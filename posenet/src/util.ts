/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

import {connectedPartIndices, angledPartIndices} from './keypoints';
import {OutputStride} from './mobilenet';
import {Keypoint, Pose, TensorBuffer3D, Vector2D} from './types';

console.log("PartIndices: ", angledPartIndices);
/*
  0: (3) [9, 7, 5] --> <wrist,emb,shoul>
  1: (3) [7, 5, 11] --> <elb, shl, hip>
  2: (3) [5, 11, 13] --> <shld, hip, knee>
  3: (3) [11, 13, 15] --> hip, knee, ank>
  4: (3) [10, 8, 6] -----> repeat
  5: (3) [8, 6, 12]
  6: (3) [6, 12, 14]
  7: (3) [12, 14, 16]
*/

function eitherPointDoesntMeetConfidence(
    a: number, b: number, minConfidence: number): boolean {
  return (a < minConfidence || b < minConfidence);
}

function anyPointDoesntMeetConfidence(
    a: number, b: number, c: number, minConfidence: number): boolean {
  return (a < minConfidence || b < minConfidence || c < minConfidence);
}

export function getAdjacentKeyPoints(
    keypoints: Keypoint[], minConfidence: number): Keypoint[][] {
  return connectedPartIndices.reduce(
      (result: Keypoint[][], [leftJoint, rightJoint]): Keypoint[][] => {
        if (eitherPointDoesntMeetConfidence(
                keypoints[leftJoint].score, keypoints[rightJoint].score,
                minConfidence)) {
          return result;
        }

        result.push([keypoints[leftJoint], keypoints[rightJoint]]);

        return result;
      }, []);
}


export function getAngledKeyPoints(
    keypoints: Keypoint[], minConfidence: number): Keypoint[][] {
  return angledPartIndices.reduce(
    (result: Keypoint[][], [jointA, jointB, jointC]): Keypoint[][] => {
      if (anyPointDoesntMeetConfidence(
        keypoints[jointA].score, keypoints[jointB].score,
        keypoints[jointC].score, minConfidence)) {
        return result;
      }

      result.push([keypoints[jointA], keypoints[jointB], keypoints[jointC]]);

      return result;
    }, []);
}

const {NEGATIVE_INFINITY, POSITIVE_INFINITY} = Number;
export function getBoundingBox(keypoints: Keypoint[]):
    {maxX: number, maxY: number, minX: number, minY: number} {
  return keypoints.reduce(({maxX, maxY, minX, minY}, {position: {x, y}}) => {
    return {
      maxX: Math.max(maxX, x),
      maxY: Math.max(maxY, y),
      minX: Math.min(minX, x),
      minY: Math.min(minY, y)
    };
  }, {
    maxX: NEGATIVE_INFINITY,
    maxY: NEGATIVE_INFINITY,
    minX: POSITIVE_INFINITY,
    minY: POSITIVE_INFINITY
  });
}

export function getBoundingBoxPoints(keypoints: Keypoint[]): Vector2D[] {
  const {minX, minY, maxX, maxY} = getBoundingBox(keypoints);
  return [
    {x: minX, y: minY}, {x: maxX, y: minY}, {x: maxX, y: maxY},
    {x: minX, y: maxY}
  ];
}

export async function toTensorBuffer<rank extends tf.Rank>(
    tensor: tf.Tensor<rank>,
    type: 'float32'|'int32' = 'float32'): Promise<tf.TensorBuffer<rank>> {
  const tensorData = await tensor.data();

  return new tf.TensorBuffer<rank>(tensor.shape, type, tensorData);
}

export async function toTensorBuffers3D(tensors: tf.Tensor3D[]):
    Promise<TensorBuffer3D[]> {
  return Promise.all(tensors.map(tensor => toTensorBuffer(tensor, 'float32')));
}

export function scalePose(pose: Pose, scaleX: number, scaleY: number): Pose {
  return {
    score: pose.score,
    keypoints: pose.keypoints.map(
        ({score, part, position}) => ({
          score,
          part,
          position: {x: position.x * scaleX, y: position.y * scaleY}
        }))
  };
}

export function scalePoses(
    poses: Pose[], scaleY: number, scaleX: number): Pose[] {
  if (scaleX === 1 && scaleY === 1) {
    return poses;
  }
  return poses.map(pose => scalePose(pose, scaleX, scaleY));
}

export function getValidResolution(
    imageScaleFactor: number, inputDimension: number,
    outputStride: OutputStride): number {
  const evenResolution = inputDimension * imageScaleFactor - 1;

  return evenResolution - (evenResolution % outputStride) + 1;
}
