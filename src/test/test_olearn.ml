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

let test_predict test_ctxt =
  let m = { theta = Vec.make 2 1.; } in
  assert_equal 2. (predict m (Vec.make 2 1.))

let test_regression test_ctxt =
  let num_features = 20 in
  let num_samples = 1000 in
  let epoch_0 = {
      h = { eta = 0.01;
            epochs = 5;
            eta_schedule = Constant;
            power_t = 0.25;
            lambda = 0.0001;
            regularization = No_regularization;
            time_steps = 1; };
      m = { theta = Vec.make0 num_features; }
  } in
  let m_true = { theta = Vec.make num_features 0.5; } in
  let rnd_state = Random.State.make_self_init () in
  let xs = Mat.random ~rnd_state:rnd_state ~from:(-5.) ~range:10. num_features num_samples in
  let xs_as_cols = Mat.to_col_vecs xs in
  let noise = Vec.random ~from:(-0.5) ~range:1. num_samples in
  let ys_no_noise = Vec.of_array (Array.map (predict m_true) xs_as_cols) in
  let ys = Vec.add ys_no_noise noise in
  let make_sample x y = { x = x; y = y } in
  let samples = BatArray.map2 make_sample xs_as_cols (Vec.to_array ys) in
  let fits = fit_regressor epoch_0 samples in
  let m_hat = (List.hd fits).m in
  let y_hats = Vec.of_array (Array.map (predict m_hat) xs_as_cols) in
  let score = r2_score ys y_hats in
  assert_bool "Bad model quality" (score > 0.97)

let test_inverse_scaling test_ctxt =
  let num_features = 200 in
  let num_samples = 10000 in
  let epoch_0 = {
      h = { eta = 0.01;
            epochs = 5;
            eta_schedule = Inverse_scaling;
            power_t = 0.25;
            lambda = 0.0001;
            regularization = No_regularization;
            time_steps = 1; };
      m = { theta = Vec.make0 num_features; }
  } in
  let m_true = { theta = Vec.make num_features 0.5; } in
  let rnd_state = Random.State.make_self_init () in
  let xs = Mat.random ~rnd_state:rnd_state ~from:(-5.) ~range:10. num_features num_samples in
  let xs_as_cols = Mat.to_col_vecs xs in
  let noise = Vec.random ~from:(-0.5) ~range:1. num_samples in
  let ys_no_noise = Vec.of_array (Array.map (predict m_true) xs_as_cols) in
  let ys = Vec.add ys_no_noise noise in
  let make_sample x y = { x = x; y = y } in
  let samples = BatArray.map2 make_sample xs_as_cols (Vec.to_array ys) in
  let fits = fit_regressor epoch_0 samples in
  let m_hat = (List.hd fits).m in
  let y_hats = Vec.of_array (Array.map (predict m_hat) xs_as_cols) in
  let score = r2_score ys y_hats in
  assert_bool "Bad model quality" (score > 0.97)

let test_L2_regularization test_ctxt =
  let num_features = 200 in
  let num_samples = 10000 in
  let epoch_0 = {
      h = { eta = 0.01;
            epochs = 5;
            eta_schedule = Inverse_scaling;
            power_t = 0.25;
            lambda = 0.0001;
            regularization = L2;
            time_steps = 1; };
      m = { theta = Vec.make0 num_features; }
  } in
  let m_true = { theta = Vec.make num_features 0.5; } in
  let rnd_state = Random.State.make_self_init () in
  let xs = Mat.random ~rnd_state:rnd_state ~from:(-5.) ~range:10. num_features num_samples in
  let xs_as_cols = Mat.to_col_vecs xs in
  let noise = Vec.random ~from:(-0.5) ~range:1. num_samples in
  let ys_no_noise = Vec.of_array (Array.map (predict m_true) xs_as_cols) in
  let ys = Vec.add ys_no_noise noise in
  let make_sample x y = { x = x; y = y } in
  let samples = BatArray.map2 make_sample xs_as_cols (Vec.to_array ys) in
  let fits = fit_regressor epoch_0 samples in
  let m_hat = (List.hd fits).m in
  let y_hats = Vec.of_array (Array.map (predict m_hat) xs_as_cols) in
  let score = r2_score ys y_hats in
  assert_bool "Bad model quality" (score > 0.97)

let suite =
  "suite" >:::
    [ "test_predict" >:: test_predict;
      "test_regression" >:: test_regression;
      "test_inverse_scaling" >:: test_inverse_scaling;
      "test_L2_regularization" >:: test_L2_regularization;
    ]

let () =
  run_test_tt_main suite
