/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as posenet from '@tensorflow-models/posenet';
import dat from 'dat.gui';
import Stats from 'stats.js';
import { drawKeypoints, drawAndGetAngles, drawSkeleton, drawBoundingBox } from './demo_util';

const videoWidth = 640;//853;//600;
const videoHeight = 360;//480;//500;

const camVidWidth = 640;
const camVidHeight = 360;
const stats = new Stats();

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupVideo() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  // const mobile = isMobile();
  // const stream = await navigator.mediaDevices.getUserMedia({
  //   'audio': false,
  //   'video': {
  //     facingMode: 'user',
  //     width: mobile ? undefined : videoWidth,
  //     height: mobile ? undefined : videoHeight,
  //   },
  // });

  const myMediaSource = new MediaSource();
  console.log(myMediaSource.readyState); // closed
  const url = URL.createObjectURL(myMediaSource);
  console.log("url initiated.")

  // video.srcObject = stream;
  video.src = url;
  myMediaSource.addEventListener('sourceopen', function (_) {
    console.log("Source opened, can procede with adding video source")

    const videoSourceBuffer = myMediaSource
      .addSourceBuffer('video/mp4; codecs="avc1.4D401F"');
    fetch("dance3_frag_silent.mp4").then(function (response) {
      // The data has to be a JavaScript ArrayBuffer
      console.log("video fetched")
      videoSourceBuffer.addEventListener('updateend', function (_) {
        myMediaSource.endOfStream();
        //video.play();
        //console.log(mediaSource.readyState); // ended
      });
      return response.arrayBuffer();
    }).then(function (videoData) {
      videoSourceBuffer.appendBuffer(videoData);
    });
  })


  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      console.log("we resolved our request i.e. video is now loaded this async op is complete.")
      resolve(video);
    };
  });
}

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video-camera');
  video.width = camVidWidth; //480;
  video.height = camVidHeight; //360;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });

  video.srcObject = stream

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      console.log("we resolved our request i.e. video is now loaded this async op is complete.")
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupVideo();
  //const videoCam = await setupCamera();
  video.play();
  //videoCam.play()

  return video;
}

async function loadVideoCam() {
  const videoCam = await setupCamera();
  //const videoCam = await setupCamera();
  videoCam.play();
  //videoCam.play()

  return videoCam;
}

const guiState = {
  algorithm: 'single-pose',
  input: {
    mobileNetArchitecture: isMobile() ? '0.50' : '0.75',
    outputStride: 16,
    imageScaleFactor: 0.5,
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showBoundingBox: false,
  },
  net: null,
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new dat.GUI({ width: 300, closed: true });

  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose works
  // for more than 1 person
  const algorithmController =
    gui.add(guiState, 'algorithm', ['single-pose']);

  // The input parameters have the most effect on accuracy and speed of the
  // network
  let input = gui.addFolder('Input');
  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  const architectureController = input.add(
    guiState.input, 'mobileNetArchitecture',
    ['1.01', '1.00', '0.75', '0.50']);
  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  input.add(guiState.input, 'outputStride', [8, 16, 32]);
  // Image scale factor: What to scale the image by before feeding it through
  // the network.
  input.add(guiState.input, 'imageScaleFactor').min(0.2).max(1.0);
  //input.open();

  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  let single = gui.addFolder('Single Pose Detection');
  single.add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0);
  single.add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0);

  // let multi = gui.addFolder('Multi Pose Detection');
  // multi.add(guiState.multiPoseDetection, 'maxPoseDetections')
  //   .min(1)
  //   .max(20)
  //   .step(1);
  // multi.add(guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0);
  // multi.add(guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0);
  // // nms Radius: controls the minimum distance between poses that are returned
  // // defaults to 20, which is probably fine for most use cases
  // multi.add(guiState.multiPoseDetection, 'nmsRadius').min(0.0).max(40.0);
  // multi.open();

  let output = gui.addFolder('Output');
  output.add(guiState.output, 'showVideo');
  output.add(guiState.output, 'showSkeleton');
  output.add(guiState.output, 'showPoints');
  output.add(guiState.output, 'showBoundingBox');
  //output.open();


  architectureController.onChange(function (architecture) {
    guiState.changeToArchitecture = architecture;
  });

  algorithmController.onChange(function (value) {
    switch (guiState.algorithm) {
      case 'single-pose':
        //multi.close();
        single.open();
        break;
      case 'multi-pose':
        single.close();
        //multi.open();
        break;
    }
  });

  gui.close();
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);
}


function startCountDown() {
  const secRemainElem = document.getElementById("master-sec-elap");
  const progBar = document.querySelector("#master-progress");
  progBar.MaterialProgress.setProgress(100);
  const maxSeconds = 60;
  secRemainElem.innerHTML = maxSeconds;
  console.log(progBar.MaterialProgress);

  var mainTimer = setInterval(function () {
    secRemainElem.innerHTML = parseInt(secRemainElem.innerHTML) - 1;
    progBar.MaterialProgress.setProgress(parseInt((maxSeconds - parseInt(secRemainElem.innerHTML)) * (100 / maxSeconds)));
  }, 1000);

  setTimeout(function () {
    clearInterval(mainTimer);
    document.getElementById('video').pause();
    document.getElementById('video-camera').pause();
    document.getElementById('output').style.display = "none";
    document.getElementById('output2').style.display = "none";
    document.getElementById('final-text').innerHTML = "Game Over! Refresh your page to try again!";
  }, (maxSeconds + 0.5) * 1000)
}

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(data) {
  console.log(data[0]);
  var video = data[0].vid;
  var net = data[0].net;
  var canvasID = data[0].cId;
  var canvWidth = data[0].vW;
  var canvHeight = data[0].vH;
  var flipHorizontal = data[0].fH;

  //console.log("cVas", canvasID);


  data = data.map(elemSet => {
    elemSet['canvas'] = document.getElementById(elemSet.cId);
    elemSet.canvas.width = elemSet.vW;
    elemSet.canvas.height = elemSet.vH;
    elemSet['ctx'] = elemSet.canvas.getContext('2d');
    return elemSet;
  });

  //console.log("NEW DATASET");
  //console.log(data);
  // since images are being fed from a webcam
  //const flipHorizontal = flipHorizontal;
  //console.log("FLIP-HORZ", flipHorizontal)

  const scoreValElem = document.getElementById("master-score-val");
  const secRemainElem = document.getElementById("master-sec-elap");


  // loop for every frame.
  async function poseDetectionFrame() {
    if (guiState.changeToArchitecture) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();

      // Load the PoseNet model weights for either the 0.50, 0.75, 1.00, or 1.01
      // version
      guiState.net = await posenet.load(+guiState.changeToArchitecture);

      guiState.changeToArchitecture = null;
    }

    // Begin monitoring code for frames per second
    stats.begin();

    // Scale an image down to a certain factor. Too large of an image will slow
    // down the GPU
    const imageScaleFactor = guiState.input.imageScaleFactor;
    const outputStride = +guiState.input.outputStride;


    let minPoseConfidence;
    let minPartConfidence;

    for (const elemSet of data) {
      let poses = [];
      const pose = await guiState.net.estimateSinglePose(
        elemSet.vid, imageScaleFactor, elemSet.fH, outputStride);
      poses.push(pose);

      minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
      minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;

      elemSet.ctx.clearRect(0, 0, elemSet.vW, elemSet.vH);

      if (guiState.output.showVideo) {
        elemSet.ctx.save();
        elemSet.ctx.scale((elemSet.fH ? -1 : 1), 1);
        elemSet.ctx.translate((elemSet.fH ? -elemSet.vW : 0), 0);
        elemSet.ctx.drawImage(elemSet.vid, 0, 0, elemSet.vW, elemSet.vH);
        elemSet.ctx.restore();
      }

      // For each pose (i.e. person) detected in an image, loop through the poses
      // and draw the resulting skeleton and keypoints if over certain confidence
      // scores
      poses.forEach(({ score, keypoints }) => {
        if (score >= minPoseConfidence) {
          if (guiState.output.showPoints) {
            drawKeypoints(keypoints, minPartConfidence, elemSet.ctx);
          }
          if (guiState.output.showSkeleton) {
            drawSkeleton(keypoints, minPartConfidence, elemSet.ctx);
            // if show skeleton also show angles
            // after drawing angles also return them out
            elemSet['angleObjArr'] = drawAndGetAngles(keypoints, minPartConfidence, elemSet.ctx);
          }
          if (guiState.output.showBoundingBox) {
            drawBoundingBox(keypoints, elemSet.ctx);
          }
        }
      });
    }



    var score = 0;

    if (data[1].angleObjArr !== undefined) {
      //make this a reduce function to get final score
      data[1].angleObjArr.forEach(angleObj => {
        var idx = data[0].angleObjArr.findIndex(x => x.name === angleObj.name);
        if (idx > -1) {
          // essentially for now we can give a 0 or 0.5 or 1 as score, maybe make it scaled.
          // max score should be 8 for 2 sets of 4 angles: alpha, beta, gamma, sigma
          // this is normalised so x/8 so u get score btw 0 -> 1
          // score is added to total

          //const targetAngle = data[0].angleObjArr[idx].angle;
          const absDiff = Math.abs(data[0].angleObjArr[idx].angle - angleObj.angle);

          score += Math.exp(-0.8 * absDiff);

        }
      });
    }


    score = score / 8; //normalised
    scoreValElem.innerHTML = (parseFloat(scoreValElem.innerHTML) + score).toFixed(2);

    // End monitoring code for frames per second
    stats.end();

    //recursive loop
    if (parseInt(secRemainElem.innerHTML) > 0) {
      //only continue detection if time is not over.
      requestAnimationFrame(poseDetectionFrame);
    }
  }

  poseDetectionFrame();
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  // Load the PoseNet model weights with architecture 0.75
  const net = await posenet.load(0.75);

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

  let video, videoCam;

  try {
    video = await loadVideo();
    videoCam = await loadVideoCam();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
      'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  console.log("both video loaded.");
  setupGui([], net);
  setupFPS();
  let data = [
    { vid: video, net: net, cId: 'output', vW: videoWidth, vH: videoHeight, fH: false },
    { vid: videoCam, net: net, cId: 'output2', vW: camVidWidth, vH: camVidHeight, fH: true },
  ]
  startCountDown();
  detectPoseInRealTime(data);
  //detectPoseInRealTime(videoCam, net, 'output2', camVidWidth, camVidHeight, true);
}

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
