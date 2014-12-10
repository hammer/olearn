open OUnit2
open Ocopt

let test_score test_ctxt =
  let m = { theta = 1.0; beta = 1.0; } in
  assert_equal 2.0 (Ocopt.score m 1.0)

(* Simple regression example from http://stattrek.com/regression/regression-example.aspx *)
(* TODO(hammer): make this into a test *)
let test_two_passes test_ctxt =
  let initial_model = { theta = 0.6; beta = 25. } in
  let initial_hyperparameters = { learning_rate = 0.0001 } in
  let x = BatArray.enum [|95.; 85.; 80.; 70.; 60.|] in
  let y = BatArray.enum [|85.; 95.; 70.; 65.; 70.|] in
  let make_sample (x, y) = { x = x; y = y } in
  let samples = BatArray.of_enum (BatEnum.map make_sample (BatEnum.combine (x, y))) in
  let m_0 = initial_model in
  let h_0 = initial_hyperparameters in
  let shuffled_samples = shuffle samples in
  let iter_1 = single_pass [m_0] shuffled_samples h_0 in
  let iter_2 = single_pass iter_1 shuffled_samples h_0 in
  iter_2

let suite =
  "suite" >:::
    [ "test_score" >:: test_score;
    ]

let () =
  run_test_tt_main suite
