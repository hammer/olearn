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
open OUnit2
open Olearn
open Lacaml.D

let num_features = 20
let num_samples = 1000
let m_true = { theta = Vec.make num_features 0.5; }
let epoch0 = {
    h = { eta = 0.01;
          epochs = 5;
          eta_schedule = Constant;
          power_t = 0.25;
          lambda = 0.0001;
          regularization = No_regularization;
          time_steps = 1; };
    m = { theta = Vec.make0 num_features; }
}

let test_predict test_ctxt =
  let m = { theta = Vec.make 2 1.; } in
  assert_equal 2. (predict m (Vec.make 2 1.))

(* TODO(hammer):
 * * Make noise optional
 * * Allow Random.State seed to be set
 * * Make range configurable
 *)
let generate_sample num_samples m_true =
  let num_features = Vec.dim m_true.theta in
  let rnd_state = Random.State.make_self_init () in
  let xs = Mat.random ~rnd_state:rnd_state ~from:(-5.) ~range:10. num_features num_samples in
  let xs_as_cols = Mat.to_col_vecs xs in
  let noise = Vec.random ~from:(-0.5) ~range:1. num_samples in
  let ys_no_noise = Vec.of_array (Array.map (predict m_true) xs_as_cols) in
  let ys = Vec.add ys_no_noise noise in
  let make_sample x y = { x = x; y = y } in
  let samples = BatArray.map2 make_sample xs_as_cols (Vec.to_array ys) in
  samples

let score_model m_hat samples =
  let y_hats = Vec.of_array (Array.map (fun s -> predict m_hat s.x) samples) in
  let ys = Vec.init (Array.length samples - 1) (fun i -> samples.(i-1).y) in
  let score = r2_score ys y_hats in
  score

let test_regression_base num_features num_samples m_true epoch0 =
  let samples = generate_sample num_samples m_true in
  let fits = fit_regressor epoch0 samples in
  let m_hat = (List.hd fits).m in
  let score = score_model m_hat samples in
  assert_bool "Bad model quality" (score > 0.9)

let test_simple_regression test_ctxt =
  test_regression_base num_features num_samples m_true epoch0

let test_inverse_scaling test_ctxt =
  let new_h = { epoch0.h with eta_schedule = Inverse_scaling } in
  let new_epoch0 = { epoch0 with h = new_h } in
  test_regression_base num_features num_samples m_true new_epoch0

let test_L2_regularization test_ctxt =
  let new_h = { epoch0.h with regularization = L2 } in
  let new_epoch0 = { epoch0 with h = new_h } in
  test_regression_base num_features num_samples m_true new_epoch0

let suite =
  "suite" >:::
    [ "test_predict" >:: test_predict;
      "test_simple_regression" >:: test_simple_regression;
      "test_inverse_scaling" >:: test_inverse_scaling;
      "test_L2_regularization" >:: test_L2_regularization;
    ]

let () =
  run_test_tt_main suite
