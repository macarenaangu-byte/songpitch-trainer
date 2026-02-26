const fs = require('fs-extra');
const path = require('path');
const { Essentia, EssentiaWASM } = require('essentia.js');
const ffmpeg = require('ffmpeg-static');
const { execSync } = require('child_process');

// Initialize Essentia
const essentia = new Essentia(EssentiaWASM);

async function decodeAndExtract(filePath) {
    // 1. Convert to 16kHz Mono WAV (Standard for AI)
    const tempWav = filePath.replace(path.extname(filePath), '.temp.wav');
    execSync(`"${ffmpeg}" -i "${filePath}" -ar 16000 -ac 1 -f wav "${tempWav}" -y`);
    
    // 2. Load the raw data
    const buffer = fs.readFileSync(tempWav);
    const audioData = new Float32Array(buffer.buffer);
    
    // 3. Extract Mel Bands (This is what the model "sees")
    // We'll take a slice of the audio to keep data sizes manageable
    const features = essentia.MelBands(essentia.arrayToVector(audioData)).mag;
    
    // Cleanup
    fs.removeSync(tempWav);
    return features;
}

async function run() {
    const datasetPath = path.join(__dirname, 'my-dataset');
    const folders = await fs.readdir(datasetPath);
    const results = [];

    for (const mood of folders) {
        if (mood === '.DS_Store') continue; // Skip Mac system files
        
        const moodPath = path.join(datasetPath, mood);
        const files = await fs.readdir(moodPath);

        for (const file of files) {
            if (file.endsWith('.mp3') || file.endsWith('.wav')) {
                console.log(`Analyzing ${mood}: ${file}...`);
                const features = await decodeAndExtract(path.join(moodPath, file));
                results.push({ label: mood, data: Array.from(features) });
            }
        }
    }

    await fs.writeJson('training-data.json', results);
    console.log('Done! training-data.json is ready.');
}

run().catch(console.error);