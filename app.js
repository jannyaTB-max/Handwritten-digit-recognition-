/* ============================================
   Neural Digit — Application Logic
   ============================================ */

// ─── MNIST Data Loader ──────────────────────────
const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

const IMAGE_SIZE = 784;       // 28 × 28
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = 10000;

class MnistData {
    constructor() {
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
    }

    async load(onProgress) {
        onProgress?.('Downloading MNIST images…');

        const img = new Image();
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        const imgRequest = new Promise((resolve) => {
            img.crossOrigin = '';
            img.onload = () => {
                img.width = img.naturalWidth;
                img.height = img.naturalHeight;

                const datasetBytesBuffer =
                    new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
                const chunkSize = 5000;
                canvas.width = img.width;
                canvas.height = chunkSize;

                for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
                    const datasetBytesView = new Float32Array(
                        datasetBytesBuffer,
                        i * IMAGE_SIZE * chunkSize * 4,
                        IMAGE_SIZE * chunkSize
                    );
                    ctx.drawImage(
                        img,
                        0, i * chunkSize, img.width, chunkSize,
                        0, 0, img.width, chunkSize
                    );
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    for (let j = 0; j < imageData.data.length / 4; j++) {
                        datasetBytesView[j] = imageData.data[j * 4] / 255;
                    }
                }
                this.datasetImages = new Float32Array(datasetBytesBuffer);
                resolve();
            };
            img.src = MNIST_IMAGES_SPRITE_PATH;
        });

        onProgress?.('Downloading MNIST labels…');
        const labelsRequest = fetch(MNIST_LABELS_PATH);
        const [, labelsResponse] = await Promise.all([imgRequest, labelsRequest]);
        this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

        this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
        this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

        this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
        this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
        this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
        this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);

        onProgress?.('MNIST data loaded ✓');
    }

    nextTrainBatch(batchSize) {
        return this.nextBatch(
            batchSize,
            [this.trainImages, this.trainLabels],
            () => {
                this.shuffledTrainIndex =
                    (this.shuffledTrainIndex + 1) % this.trainIndices.length;
                return this.trainIndices[this.shuffledTrainIndex];
            }
        );
    }

    nextTestBatch(batchSize) {
        return this.nextBatch(
            batchSize,
            [this.testImages, this.testLabels],
            () => {
                this.shuffledTestIndex =
                    (this.shuffledTestIndex + 1) % this.testIndices.length;
                return this.testIndices[this.shuffledTestIndex];
            }
        );
    }

    nextBatch(batchSize, data, index) {
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
        const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

        for (let i = 0; i < batchSize; i++) {
            const idx = index();
            const image = data[0].slice(
                idx * IMAGE_SIZE,
                idx * IMAGE_SIZE + IMAGE_SIZE
            );
            batchImagesArray.set(image, i * IMAGE_SIZE);

            const label = data[1].slice(
                idx * NUM_CLASSES,
                idx * NUM_CLASSES + NUM_CLASSES
            );
            batchLabelsArray.set(label, i * NUM_CLASSES);
        }

        const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
        const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);
        return { xs, labels };
    }
}

// ─── CNN Model ──────────────────────────────────
function createModel() {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({
        units: 10,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

// ─── Global State ───────────────────────────────
let model = null;
let isTraining = false;
let hasDrawn = false;

// ─── DOM References ─────────────────────────────
const drawCanvas = document.getElementById('drawCanvas');
const drawCtx = drawCanvas.getContext('2d');
const previewCanvas = document.getElementById('previewCanvas');
const previewCtx = previewCanvas.getContext('2d');
const canvasOverlay = document.getElementById('canvasOverlay');

const clearBtn = document.getElementById('clearBtn');
const trainBtn = document.getElementById('trainBtn');
const brushSizeSlider = document.getElementById('brushSize');
const brushSizeLabel = document.getElementById('brushSizeLabel');

const modelStatusBadge = document.getElementById('modelStatus');
const statusText = modelStatusBadge.querySelector('.status-text');

const predictedDigit = document.getElementById('predictedDigit');
const predictedConfidence = document.getElementById('predictedConfidence');

const trainingProgress = document.getElementById('trainingProgress');
const progressLabel = document.getElementById('progressLabel');
const progressPercent = document.getElementById('progressPercent');
const progressBarFill = document.getElementById('progressBarFill');
const statLoss = document.getElementById('statLoss');
const statAccuracy = document.getElementById('statAccuracy');
const statEpoch = document.getElementById('statEpoch');
const statTestAcc = document.getElementById('statTestAcc');

const epochsInput = document.getElementById('epochsInput');
const batchInput = document.getElementById('batchInput');

// ─── Canvas Setup ───────────────────────────────
function initCanvas() {
    drawCtx.fillStyle = '#000000';
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    drawCtx.strokeStyle = '#ffffff';
    drawCtx.lineWidth = parseInt(brushSizeSlider.value);
    drawCtx.lineCap = 'round';
    drawCtx.lineJoin = 'round';
}
initCanvas();

// ─── Drawing Logic ──────────────────────────────
let isDrawing = false;
let lastX = 0;
let lastY = 0;

function getCanvasPos(e) {
    const rect = drawCanvas.getBoundingClientRect();
    const scaleX = drawCanvas.width / rect.width;
    const scaleY = drawCanvas.height / rect.height;

    if (e.touches && e.touches.length > 0) {
        return {
            x: (e.touches[0].clientX - rect.left) * scaleX,
            y: (e.touches[0].clientY - rect.top) * scaleY
        };
    }
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    const pos = getCanvasPos(e);
    lastX = pos.x;
    lastY = pos.y;

    // Draw a dot for single clicks
    drawCtx.beginPath();
    drawCtx.arc(lastX, lastY, drawCtx.lineWidth / 2, 0, Math.PI * 2);
    drawCtx.fillStyle = '#ffffff';
    drawCtx.fill();

    if (!hasDrawn) {
        hasDrawn = true;
        canvasOverlay.classList.add('hidden');
    }
}

function draw(e) {
    e.preventDefault();
    if (!isDrawing) return;

    const pos = getCanvasPos(e);
    drawCtx.beginPath();
    drawCtx.moveTo(lastX, lastY);
    drawCtx.lineTo(pos.x, pos.y);
    drawCtx.stroke();
    lastX = pos.x;
    lastY = pos.y;

    if (model) requestAnimationFrame(() => predict());
}

function stopDrawing(e) {
    if (e) e.preventDefault();
    if (!isDrawing) return;
    isDrawing = false;
    if (model && hasDrawn) predict();
}

// Mouse events
drawCanvas.addEventListener('mousedown', startDrawing);
drawCanvas.addEventListener('mousemove', draw);
drawCanvas.addEventListener('mouseup', stopDrawing);
drawCanvas.addEventListener('mouseleave', stopDrawing);

// Touch events
drawCanvas.addEventListener('touchstart', startDrawing, { passive: false });
drawCanvas.addEventListener('touchmove', draw, { passive: false });
drawCanvas.addEventListener('touchend', stopDrawing, { passive: false });
drawCanvas.addEventListener('touchcancel', stopDrawing, { passive: false });

// ─── Brush Size ─────────────────────────────────
brushSizeSlider.addEventListener('input', () => {
    const size = parseInt(brushSizeSlider.value);
    drawCtx.lineWidth = size;
    brushSizeLabel.textContent = size + 'px';
});

// ─── Clear Canvas ───────────────────────────────
clearBtn.addEventListener('click', () => {
    drawCtx.fillStyle = '#000000';
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    hasDrawn = false;
    canvasOverlay.classList.remove('hidden');

    // Reset predictions
    predictedDigit.textContent = '?';
    predictedConfidence.textContent = '—';
    predictedDigit.classList.remove('bounce');

    // Clear preview
    previewCtx.fillStyle = '#000000';
    previewCtx.fillRect(0, 0, 28, 28);

    // Reset probability bars
    document.querySelectorAll('.prob-row').forEach((row) => {
        row.classList.remove('highlight');
        row.querySelector('.prob-fill').style.width = '0%';
        row.querySelector('.prob-value').textContent = '0%';
    });
});

// ─── Preprocessing ──────────────────────────────

/**
 * Converts a region of the drawing canvas [x, y, w, h] into a
 * 28×28 MNIST-normalised Float32Array (grayscale 0–1).
 */
function regionToMnistInput(srcCanvas, rx, ry, rw, rh) {
    // Crop the region
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = rw;
    cropCanvas.height = rh;
    const cropCtx = cropCanvas.getContext('2d');
    cropCtx.drawImage(srcCanvas, rx, ry, rw, rh, 0, 0, rw, rh);

    // Scale to fit inside a 20×20 box preserving aspect ratio (MNIST convention)
    const scale = 20 / Math.max(rw, rh);
    const scaledW = Math.max(1, Math.round(rw * scale));
    const scaledH = Math.max(1, Math.round(rh * scale));

    // Centre in 28×28
    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = 28;
    finalCanvas.height = 28;
    const finalCtx = finalCanvas.getContext('2d');
    finalCtx.fillStyle = '#000000';
    finalCtx.fillRect(0, 0, 28, 28);

    const offsetX = Math.round((28 - scaledW) / 2);
    const offsetY = Math.round((28 - scaledH) / 2);
    finalCtx.imageSmoothingEnabled = true;
    finalCtx.imageSmoothingQuality = 'medium';
    finalCtx.drawImage(cropCanvas, 0, 0, rw, rh, offsetX, offsetY, scaledW, scaledH);

    const finalImgData = finalCtx.getImageData(0, 0, 28, 28);
    const grayscale = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
        grayscale[i] = finalImgData.data[i * 4] / 255;
    }
    return { grayscale, finalCanvas };
}

/**
 * Analyses the drawing canvas and returns an array of digit segments.
 * Each segment: { x, y, w, h } in canvas coordinates.
 *
 * Strategy:
 *  1. Find the overall bounding box of all drawn pixels.
 *  2. Build a column-wise ink profile (sum of bright pixels per column).
 *  3. Detect vertical gap columns (no ink) to split into individual digits.
 *  4. Merge segments that are too narrow (stray marks / diacritics).
 */
function segmentDigits() {
    const W = drawCanvas.width;
    const H = drawCanvas.height;
    const imgData = drawCtx.getImageData(0, 0, W, H);
    const pixels = imgData.data;

    // ── 1. Overall bounding box ──────────────────
    let gMinX = W, gMaxX = 0, gMinY = H, gMaxY = 0;
    let hasContent = false;

    const colHasInk = new Uint8Array(W); // 1 if column has any bright pixel

    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const idx = (y * W + x) * 4;
            if (pixels[idx] > 10 || pixels[idx + 1] > 10 || pixels[idx + 2] > 10) {
                hasContent = true;
                colHasInk[x] = 1;
                if (x < gMinX) gMinX = x;
                if (x > gMaxX) gMaxX = x;
                if (y < gMinY) gMinY = y;
                if (y > gMaxY) gMaxY = y;
            }
        }
    }

    if (!hasContent) return null;

    // ── 2. Detect column gaps inside the bounding box ──
    // A "gap" column is one with no ink.
    const segments = [];
    let segStart = -1;

    for (let x = gMinX; x <= gMaxX + 1; x++) {
        const ink = (x <= gMaxX) ? colHasInk[x] : 0;
        if (ink && segStart === -1) {
            segStart = x; // start of a new digit segment
        } else if (!ink && segStart !== -1) {
            segments.push({ x: segStart, w: x - segStart });
            segStart = -1;
        }
    }

    // ── 3. Add vertical extent + padding ────────
    const vPad = Math.round(H * 0.08);  // 8% vertical padding
    const hPad = Math.round(W * 0.03);  // 3% horizontal padding
    const minSegW = 8; // ignore slivers narrower than 8 px

    const result = [];
    for (const seg of segments) {
        if (seg.w < minSegW) continue;

        // Find vertical extent of this segment's columns
        let sMinY = H, sMaxY = 0;
        for (let x = seg.x; x < seg.x + seg.w; x++) {
            for (let y = 0; y < H; y++) {
                const idx = (y * W + x) * 4;
                if (pixels[idx] > 10 || pixels[idx + 1] > 10 || pixels[idx + 2] > 10) {
                    if (y < sMinY) sMinY = y;
                    if (y > sMaxY) sMaxY = y;
                }
            }
        }

        const rx = Math.max(0, seg.x - hPad);
        const ry = Math.max(0, sMinY - vPad);
        const rx2 = Math.min(W - 1, seg.x + seg.w - 1 + hPad);
        const ry2 = Math.min(H - 1, sMaxY + vPad);

        result.push({ x: rx, y: ry, w: rx2 - rx + 1, h: ry2 - ry + 1 });
    }

    return result.length > 0 ? result : null;
}

// ─── Prediction ─────────────────────────────────
let lastPredicted = '';

function predict() {
    if (!model) return;

    const segments = segmentDigits();
    if (!segments) return;

    // ── Build batch of all digit segments ──────
    const batchSize = segments.length;
    const batchArray = new Float32Array(batchSize * 784);
    const previews = [];

    segments.forEach((seg, i) => {
        const { grayscale, finalCanvas } = regionToMnistInput(
            drawCanvas, seg.x, seg.y, seg.w, seg.h
        );
        batchArray.set(grayscale, i * 784);
        previews.push(finalCanvas);
    });

    // Show the first segment in the preview canvas (left-most digit)
    previewCtx.fillStyle = '#000';
    previewCtx.fillRect(0, 0, 28, 28);
    previewCtx.drawImage(previews[0], 0, 0);

    tf.tidy(() => {
        const tensor = tf.tensor4d(batchArray, [batchSize, 28, 28, 1]);
        const predTensor = model.predict(tensor);
        const allProbs = predTensor.arraySync(); // [batchSize][10]

        // Build the recognised number string
        const digits = allProbs.map(probs => {
            let maxIdx = 0;
            for (let i = 1; i < 10; i++) {
                if (probs[i] > probs[maxIdx]) maxIdx = i;
            }
            return maxIdx;
        });

        const number = digits.join('');

        // Avg confidence across all digit predictions
        const avgConf = allProbs.reduce((sum, probs) => {
            const maxProb = Math.max(...probs);
            return sum + maxProb;
        }, 0) / batchSize;

        // For probability bars: use the first digit's probabilities
        const firstProbs = allProbs[0];

        updatePredictionUI(number, avgConf, firstProbs, digits[0]);
    });
}

function updatePredictionUI(number, avgConf, firstProbs, firstDigit) {
    // Animate if result changed
    if (number !== lastPredicted) {
        predictedDigit.classList.remove('bounce');
        void predictedDigit.offsetWidth; // force reflow
        predictedDigit.classList.add('bounce');
        lastPredicted = number;
    }
    predictedDigit.textContent = number;

    // Confidence label
    const numDigits = number.length;
    if (numDigits > 1) {
        predictedConfidence.textContent =
            `${numDigits} digits · avg ${(avgConf * 100).toFixed(1)}% confidence`;
    } else {
        predictedConfidence.textContent =
            `${(avgConf * 100).toFixed(1)}% confidence`;
    }

    // Probability bars show probabilities for the first (left-most) digit
    const rows = document.querySelectorAll('.prob-row');
    rows.forEach((row, i) => {
        const prob = firstProbs[i];
        const percent = (prob * 100).toFixed(1);
        row.querySelector('.prob-fill').style.width = percent + '%';
        row.querySelector('.prob-value').textContent = percent + '%';
        if (i === firstDigit) {
            row.classList.add('highlight');
        } else {
            row.classList.remove('highlight');
        }
    });
}

// ─── Training ───────────────────────────────────
trainBtn.addEventListener('click', async () => {
    if (isTraining) return;
    isTraining = true;

    // UI updates
    trainBtn.disabled = true;
    trainBtn.classList.add('loading');
    trainBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/>
            <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/>
            <line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/>
            <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/>
        </svg>
        Training…`;

    modelStatusBadge.className = 'header-badge training';
    statusText.textContent = 'Training…';

    trainingProgress.style.display = 'block';

    const numEpochs = parseInt(epochsInput.value);
    const batchSize = parseInt(batchInput.value);

    try {
        // Load MNIST data
        const data = new MnistData();
        await data.load((msg) => {
            progressLabel.textContent = msg;
        });

        progressLabel.textContent = 'Building model…';

        // Create model
        model = createModel();

        // Prepare training data
        const trainBatch = data.nextTrainBatch(NUM_TRAIN_ELEMENTS);
        const trainXs = trainBatch.xs.reshape([NUM_TRAIN_ELEMENTS, 28, 28, 1]);

        // Prepare test data
        const testBatch = data.nextTestBatch(NUM_TEST_ELEMENTS);
        const testXs = testBatch.xs.reshape([NUM_TEST_ELEMENTS, 28, 28, 1]);

        progressLabel.textContent = 'Training…';

        const totalBatches = Math.ceil(NUM_TRAIN_ELEMENTS / batchSize) * numEpochs;
        let batchCount = 0;

        // Train the model
        await model.fit(trainXs, trainBatch.labels, {
            batchSize: batchSize,
            epochs: numEpochs,
            validationSplit: 0.1,
            callbacks: {
                onBatchEnd: (batch, logs) => {
                    batchCount++;
                    const pct = ((batchCount / totalBatches) * 100).toFixed(0);
                    progressBarFill.style.width = pct + '%';
                    progressPercent.textContent = pct + '%';
                    statLoss.textContent = logs.loss.toFixed(4);
                    statAccuracy.textContent = (logs.acc * 100).toFixed(1) + '%';
                },
                onEpochEnd: (epoch, logs) => {
                    statEpoch.textContent = `${epoch + 1} / ${numEpochs}`;
                    progressLabel.textContent = `Epoch ${epoch + 1} / ${numEpochs}`;
                    statAccuracy.textContent = (logs.acc * 100).toFixed(1) + '%';
                    statLoss.textContent = logs.loss.toFixed(4);
                }
            }
        });

        // Evaluate on test set
        progressLabel.textContent = 'Evaluating on test set…';
        const evalResult = model.evaluate(testXs, testBatch.labels);
        const testAcc = (await evalResult[1].data())[0];
        statTestAcc.textContent = (testAcc * 100).toFixed(1) + '%';

        // Cleanup tensors
        trainXs.dispose();
        trainBatch.labels.dispose();
        trainBatch.xs.dispose();
        testXs.dispose();
        testBatch.labels.dispose();
        testBatch.xs.dispose();
        evalResult[0].dispose();
        evalResult[1].dispose();

        // Success state
        progressBarFill.style.width = '100%';
        progressPercent.textContent = '100%';
        progressLabel.textContent = `Training complete! Test accuracy: ${(testAcc * 100).toFixed(1)}%`;

        modelStatusBadge.className = 'header-badge trained';
        statusText.textContent = `Model Ready (${(testAcc * 100).toFixed(1)}%)`;

        trainBtn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"/>
            </svg>
            Retrain`;
        trainBtn.disabled = false;
        trainBtn.classList.remove('loading');

    } catch (error) {
        console.error('Training error:', error);
        progressLabel.textContent = 'Error: ' + error.message;
        modelStatusBadge.className = 'header-badge';
        statusText.textContent = 'Training Failed';

        trainBtn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
            Retry`;
        trainBtn.disabled = false;
        trainBtn.classList.remove('loading');
    }

    isTraining = false;
});

// ─── Init ───────────────────────────────────────
previewCtx.fillStyle = '#000000';
previewCtx.fillRect(0, 0, 28, 28);

console.log('🧠 Neural Digit — Ready. Click "Train Model" to begin.');
