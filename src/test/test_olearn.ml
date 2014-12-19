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

let get_rand l r =
  let spread = r -. l in
  r -. Random.float spread

let test_predict test_ctxt =
  let m = { theta = [| 1.0 |]; beta = 1.0; } in
  assert_equal 2.0 (predict m [| 1.0 |])

let test_simple_regression test_ctxt =
  let epoch_0 = {
      h = { learning_rate = 0.01; epochs = 20; };
      m = { theta = [| 0. |]; beta = 0.; }
  } in
  let xs = Array.init 100 (fun _ -> [| (get_rand (-5.) 5.) |]) in
  let ys = Array.map (fun x -> 0.5 *. x.(0) +. get_rand 0. 1.) xs in
  let make_sample x y = { x = x; y = y } in
  let samples = BatArray.map2 make_sample xs ys in
  let fits = fit_regressor epoch_0 samples in
  let m_hat = (List.hd fits).m in
  let y_hats = Array.map (fun x -> predict m_hat x) xs in
  let score = r2_score ys y_hats in
  assert_bool "Bad model quality" (score > 0.95)

let test_multivariate_regression test_ctxt =
  let num_features = 10 in
  let epoch_0 = {
      h = { learning_rate = 0.01; epochs = 20; };
      m = { theta = Array.make num_features 0.; beta = 0.; }
  } in
  let xs = Array.init 100 (fun _ -> Array.init num_features (fun _ -> get_rand (-5.) 5.)) in
  let theta = Array.init num_features (fun _ -> get_rand 0.1 0.9) in
  let y x = BatArray.fsum ((BatArray.map2 (fun a b -> a *. b) theta x)) +. get_rand 0. 1. in
  let ys = Array.map y xs in
  let make_sample x y = { x = x; y = y } in
  let samples = BatArray.map2 make_sample xs ys in
  let fits = fit_regressor epoch_0 samples in
  let m_hat = (List.hd fits).m in
  let y_hats = Array.map (fun x -> predict m_hat x) xs in
  let score = r2_score ys y_hats in
  assert_bool "Bad model quality" (score > 0.95)

let suite =
  "suite" >:::
    [ "test_predict" >:: test_predict;
      "test_simple_regression" >:: test_simple_regression;
      "test_multivariate_regression" >:: test_multivariate_regression;
    ]

let () =
  run_test_tt_main suite
