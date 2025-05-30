from statsmodels.tsa.stattools import adfuller, kpss


# ADF
def adfuller_test(data):
    adf_test = adfuller(data, autolag="AIC")  # AIC is the default option
    print("ADF Statistic:", adf_test[0])
    print("p-value: ", adf_test[1])
    print("Critical Values:")
    for key, value in adf_test[4].items():
        print("\t%s: %.3f" % (key, value))
    if adf_test[1] <= 0.05:
        print("We can reject the null hypothesis (H0) --> data is stationary")
    else:
        print("We cannot reject the null hypothesis (H0) --> data is non-stationary")


# KPSS
# Note: "regression" represents the null hypothesis for the KPSS test. There are two options:
# “c” : The data is stationary around a constant (default).
# “ct” : The data is stationary around a trend
def kpss_test(data):
    kpss_out = kpss(data, regression="c", nlags="auto", store=True)
    print("KPSS Statistic:", kpss_out[0])
    print("p-value: ", kpss_out[1])
    if kpss_out[1] <= 0.05:
        print(
            "We can reject the null hypothesis (H0) --> unit root, data is not stationary"
        )
    else:
        print("We cannot reject the null hypothesis (H0) --> data is trend stationary")
