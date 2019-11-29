## Notes about METR-LA

119 Days, 5 min/times, 34272 time slot


Model | MSE | MAE | RMSE | epoch | parameter
:-: | :-: | :-: | :-: | :-: | :-:
seq2seq_lstm | 47.9769 | 3.5181 | 6.9265 | 200 |1,053,647
seq2seq_gru | 44.6624 | 3.2838 | 6.6830 | 200 | 805,071
basic_gcn | 54.8226 | 3.9339 | 7.4042 | 200 | 685,794
system_1 | 60.0933 | 3.4950 | 7.7520 | 200 | 1,576,770
system_2 | 48.2176 | 3.3985 | 6.9439 | 200 | 890,770
seq2seq_gru(data_completion) | 18.0571 | 2.6648 | 4.2494 | 500 | 805,071
system_model | 17.2004 | 2.6051 | 4.1473 | 400 | 890,770
