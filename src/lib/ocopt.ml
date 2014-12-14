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

(* Thanks, batteries (and Fisher-Yates) *)
let shuffle a =
  for n = Array.length a - 1 downto 1 do
    let k    = Random.int ( n + 1 ) in
    if k <> n then
      let buf  = Array.get a n in
      Array.set a n (Array.get a k);
      Array.set a k buf
  done;
  a

let print_samples ss =
  Array.iter (fun s -> Printf.printf "x: %f y: %f\n" s.x s.y) ss

let print_models ms =
  List.iter (fun m -> Printf.printf "theta: %f beta: %f\n" m.theta m.beta) ms

(* TODO(hammer): verify these arrays have the same length *)
let r2_score y y_hat =
  let y_len = Array.length y in
  let diff_sq = Array.init y_len (fun i -> (y.(i) -. y_hat.(i)) ** 2.) in
  let numerator = Array.fold_left (fun x y -> x +. y) 0. diff_sq in
  let y_avg = (Array.fold_left (fun x y -> x +. y) 0. y) /. (float y_len) in
  let y_spread_sq = Array.init y_len (fun i -> (y.(i) -. y_avg) ** 2.) in
  let denominator = Array.fold_left (fun x y -> x +. y) 0. y_spread_sq in
  match () with
  | () when denominator = 0.0 && numerator = 0.0 -> 1.0
  | () when denominator = 0.0 -> 0.0
  | () -> 1. -. numerator /. denominator

let predict m x =
  m.theta *. x +. m.beta

let step_parameters m s h =
  let y_hat = predict m s.x in
  { theta = m.theta -. h.learning_rate *. (y_hat -. s.y) *. s.x;
    beta = m.beta -. h.learning_rate *. (y_hat -. s.y)
  }

let single_pass m samples h =
  let f ms s = (step_parameters (List.hd ms) s h) :: ms in
  Array.fold_left f m samples

let fit_regressor samples h =
  let m = { theta = 0.0; beta = 0.0; } in
  let ms = Array.make (h.epochs + 1) [] in
  ms.(0) <- [m];
  let shuffled_samples = shuffle samples in
  for i = 1 to h.epochs do
    ms.(i) <- single_pass ms.(i-1) shuffled_samples h
  done;
  ms
