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

let test_predict test_ctxt =
  let m = { theta = 1.0; beta = 1.0; } in
  assert_equal 2.0 (predict m 1.0)

let test_simple_regression test_ctxt =
  let epoch_0 = {
      h = { learning_rate = 0.01; epochs = 20; };
      m = { theta = 0.; beta = 0.; }
  } in
  let xs = Array.init 100 (fun _ -> 5.0 -. Random.float 10.0) in
  let ys = Array.map (fun x -> 0.5 *. x) xs in
  let make_sample x y = { x = x; y = y } in
  let samples = BatArray.map2 make_sample xs ys in
  let fits = fit_regressor epoch_0 samples in
  let fitted_model = (List.hd fits).m in
  let y_hats = Array.map (fun x -> predict fitted_model x) xs in
  let score = r2_score ys y_hats in
  assert_bool "Bad model quality" (score > 0.99)

let suite =
  "suite" >:::
    [ "test_predict" >:: test_predict;
      "test_simple_regression" >:: test_simple_regression;
    ]

let () =
  run_test_tt_main suite
