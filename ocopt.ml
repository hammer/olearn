(* Simple regression example from http://stattrek.com/regression/regression-example.aspx *)
let x = [95; 85; 80; 70; 60]
let y = [85; 95; 70; 65; 70]

(* Initial parameter values *)
let theta = Random.float 1.0
let beta = Random.float 1.0
