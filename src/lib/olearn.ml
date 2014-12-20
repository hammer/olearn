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
open Lacaml.D

type input = Lacaml_float64.vec
type inputs = Lacaml_float64.mat
type output = float
type outputs = Lacaml_float64.vec
type predicted_output = output
type predicted_outputs = Lacaml_float64.vec
type num_epochs = int
type r2_score = float
type eta_schedule = Constant | Inverse_scaling

type sample = { x : input; y : output; }
type model = { theta : Lacaml_float64.vec; }
type hyperparameters = {
    epochs : num_epochs;
    eta : float;
    eta_schedule : eta_schedule;
    power_t : float;
    time_steps : int;
  }
type epoch_state = { m : model; h : hyperparameters; }
type samples = sample array
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
  let numerator = Vec.ssqr_diff ys y_hats in
  let y_avg = Vec.sum ys /. float (Vec.dim ys) in
  let denominator = Vec.ssqr ~c:y_avg ys in
  match numerator, denominator with
  | 0., 0. -> 1.
  | _, 0. -> 0.
  | _, _ -> 1. -. numerator /. denominator

let predict m x =
  dot m.theta x

let fit_sample epoch_state s =
  let m = epoch_state.m in
  let h = epoch_state.h in
  let y_hat = predict m s.x in
  let step = h.eta *. (y_hat -. s.y) in
  let grad = Vec.mul (Vec.make (Vec.dim s.x) step) s.x in
  let new_theta = Vec.sub m.theta grad in
  let new_eta = match h.eta_schedule with
  | Inverse_scaling -> ((float (h.time_steps) /. float (h.time_steps + 1)) ** h.power_t) *. h.eta
  | Constant -> h.eta in
  {
    m = { theta = new_theta; };
    h = { h with eta = new_eta; time_steps = h.time_steps + 1; }
  }

let fit_epoch fits samples =
  let f fits s = (fit_sample (List.hd fits) s) :: fits in
  Array.fold_left f fits samples

let rec fit_epochs remaining_epochs fits samples =
  match remaining_epochs with
  | 0 -> fits
  | _ -> fit_epochs (remaining_epochs - 1) (fit_epoch fits samples) samples

let fit_regressor epoch_0 samples =
  let shuffled_samples = shuffle samples in
  fit_epochs epoch_0.h.epochs [epoch_0] shuffled_samples
