let video, classifier, faceapi, inputLabel
let boxes = []
let trained = false
//const imgSize = 256

const w = 640;
const h = 360;


function setup() {
  createCanvas(w,h);

  initVideo()
  initFaceDetector()
  pixelDensity(1);
  //initFaceClassifier()
  //drawMenu()
}

function draw() {
 clear();
 //background(255); // this will draw nonstop
  //image(video, 0, 0)
  drawBoxes()
}

function drawBoxes() {
  for (let i=0; i < boxes.length; i++) {
    const box = boxes[i]
    noFill()
    stroke(0,255, 0)
    strokeWeight(2)
    rect(box.x, box.y, box.width, box.height)

    if (box.label) {
      fill(161, 95, 251)
      rect(box.x, box.y + box.height, 100, 25)

      fill(255)
      noStroke()
      // strokeWeight(2)
      textSize(18)
      // text(box.label, box.x + 10, box.y + box.height + 20)
    }
  }
}

function initVideo() {
  video = createCapture(VIDEO)
  video.size(w,h)
  // video.hide()
}

function initFaceClassifier() {
  let options = {
    inputs: [w/2, h/2, 4],
    task: 'imageClassification',
    debug: true,
  }
  classifier = ml5.neuralNetwork(options)
}

function initFaceDetector() {
  const detectionOptions = {
    withLandmarks: true,
    withDescriptors: false
  };

  faceapi = ml5.faceApi(video, detectionOptions, () => {
    console.log('Face API Model Loaded!')
    detectFace()
  });
}

function detectFace() {
  faceapi.detect((err, results) => {
    if (err) return console.error(err)

    boxes = []
    if (results && results.length > 0) {
      boxes = getBoxes(results)
      if (trained) {
        for (let i=0; i < boxes.length; i++) {
          const box = boxes[i]
          classifyFace(box)
        }
      }
    }
    detectFace()
  })
}

function getBoxes(detections) {
  const boxes = []
  for(let i = 0; i < detections.length; i++) {
    const alignedRect = detections[i].alignedRect

    const box = {
      x: alignedRect._box._x,
      y: alignedRect._box._y,
      width: alignedRect._box._width,
      height: alignedRect._box._height,
      label: ""
    }
    boxes.push(box)
  }

  return boxes
}

function classifyFace(box) {
  const img = get(box.x, box.y, box.width, box.height)
  img.resize(imgSize, imgSize)
  let inputImage = { image: img };
  classifier.classify(inputImage, (error, results) => {
    if (error) return console.error(error)

    // The results are in an array ordered by confidence.
    label = results[0].label
    box.label = label
  });
}
