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

let test_score test_ctxt =
  let m = { theta = 1.0; beta = 1.0; } in
  assert_equal 2.0 (predict m 1.0)

let test_simple_regression test_ctxt =
  let h = { learning_rate = 0.01; epochs = 20; } in
  let x = Array.init 100 (fun _ -> 5.0 -. Random.float 10.0) in
  let y = Array.init 100 (fun i -> 0.5 *. x.(i)) in
  let make_sample (x, y) = { x = x; y = y } in
  let samples = Array.init 100 (fun i -> make_sample (x.(i), y.(i))) in
  let ms = fit_regressor samples h in
  let fitted_model = List.hd ms.(h.epochs) in
  let y_hat = Array.init 100 (fun i -> predict fitted_model x.(i)) in
  let score = r2_score y y_hat in
  assert_bool "Bad model quality" (score > 0.99)

let suite =
  "suite" >:::
    [ "test_score" >:: test_score;
      "test_simple_regression" >:: test_score;
    ]

let () =
  run_test_tt_main suite
