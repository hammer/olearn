(**
 * Copyright 2014 Jeff Hammerbacher
 * Licensed under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *)
type sample = {
  x : float;
  y : float;
}

type model = {
  theta : float;
  beta  : float;
}

type hyperparameters = {
  epochs : int;
  learning_rate : float;
}

type epoch_state = {
  m : model;
  h : hyperparameters;
}

type fits = epoch_state list


(* Thanks, batteries (and Fisher-Yates) *)
let shuffle a =
  for n = Array.length a - 1 downto 1 do
    let k = Random.int (n + 1) in
    if k <> n then
      let buf = a.(n) in
      a.(n) <- a.(k);
      a.(k) <- buf
  done;
  a

(* TODO(hammer): verify these arrays have the same length *)
let r2_score ys y_hats =
  let diff_sq = BatArray.map2 (fun y y_hat -> (y -. y_hat) ** 2.) ys y_hats in
  let numerator = BatArray.fsum diff_sq in
  let y_avg = BatArray.favg ys in
  let y_spread_sq = BatArray.mapi (fun i y -> (y -. y_avg) ** 2.) ys in
  let denominator = BatArray.fsum y_spread_sq in
  match numerator, denominator with
  | 0.0, 0.0 -> 1.0
  | _, 0.0 -> 0.0
  | _, _ -> 1. -. numerator /. denominator

let predict m x =
  m.theta *. x +. m.beta

let step_parameters epoch_state s =
  let m = epoch_state.m in
  let h = epoch_state.h in
  let y_hat = predict m s.x in
  let new_theta = m.theta -. h.learning_rate *. (y_hat -. s.y) *. s.x in
  let new_beta = m.beta -. h.learning_rate *. (y_hat -. s.y) in
  {
    m = { theta = new_theta;
          beta = new_beta;
        };
    h = h;
  }

let single_pass fits samples =
  let f fits s = (step_parameters (List.hd fits) s) :: fits in
  Array.fold_left f fits samples

let rec all_passes remaining_epochs fits samples =
  match remaining_epochs with
  | 0 -> fits
  | _ -> all_passes (remaining_epochs - 1) (single_pass fits samples) samples

let fit_regressor samples h =
  let m = { theta = 0.0; beta = 0.0; } in
  let epoch_0 = { m = m; h = h } in
  let shuffled_samples = shuffle samples in
  all_passes h.epochs [epoch_0] shuffled_samples
