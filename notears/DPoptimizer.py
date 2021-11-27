import DPoptimizer_plain_vanilla as dpopt1
import DPoptimizer_group_clipping as dpopt2
import DPoptimizer_adaclip2 as dpopt3
import DPoptimizer_adap_quantile2 as dpopt4
import DPoptimizer_adaclip_and_adap_quantile2 as dpopt5
import DPoptimizer_group_clipping_and_adap_quantile2 as dpopt6


def make_optimizer_class(cls, method):
    if method == 'plain_vanilla':
        return dpopt1.make_optimizer_class(cls)
    elif method == 'group_clipping':
        return dpopt2.make_optimizer_class(cls)
    elif method == 'adaclip':
        return dpopt3.make_optimizer_class(cls)
    elif method == 'adap_quantile':
        return dpopt4.make_optimizer_class(cls)
    elif method == 'adaclip_and_adap_quantile':
        return dpopt5.make_optimizer_class(cls)
    elif method == 'group_clipping_and_adap_quantile':
        return dpopt6.make_optimizer_class(cls)
    else:
        print('DP optimizer method not recognized')


