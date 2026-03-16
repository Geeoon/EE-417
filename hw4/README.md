# Instructions
1. Run `pip install -r requirements.txt` to install the dependencies.
2. Run `python3 test_main.py` to run the code.  This may take a very long time (several days).
3. The resulting graphs are in the files `results/simulated.png` and `results/expected.png`

# Quantitative Analysis
The empirical results we got from the simulation are different from the expected results for 4-QAM with a coding gain of 5 (derived from our G matrix).  Although this they differ, it is to be expected.  The expected results are just the upper bound on the error rates.  So our empirical results being lower makes sense.

Between the soft and hard decoding, we can see that soft decoding has much lower error rate the hard decoding.  This also makes sense because soft decoding takes into account the actual symbols rather than collapsing the information into two bits.

Between hard and soft decoding, we observe an SNR difference of about dB.
