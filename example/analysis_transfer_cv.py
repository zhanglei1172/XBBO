import time, glob, os, yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_csv(exp_dir):
    res_l = []
    tim_l = []
    name = glob.glob(exp_dir + '/*.yaml')[0]
    with open(name, 'r') as f:
        y = yaml.load(f)

    # rank_max = len(pd.read_csv("/home/zhang/PycharmProjects/MAC/TST/data/svm/"+y['TEST_PROBLEM']['kwargs']['test_data'], sep=' ', header=None))

    for file in sorted(glob.glob(exp_dir + '/res/res*.csv')):
        res = pd.read_csv(file, index_col=[0, 1], header=[0, 1])
        res = res.groupby(by='call', axis=0).median()

        # res[('func_evals', 'rank')] += 1
        # res[('func_evals', 'rank')] /= rank_max
        # res[('func_evals', 'rank')] += 1
        # FIXME log(0) bug
        res[('func_evals', 'log_regret')] = np.log10(res[('func_evals', 'regret')]+0.01)
        # res[('best_func_evals', 'best_rank')] = res[('func_evals', 'rank')].cummin()
        res[('best_func_evals', 'best_regret')] = res[('func_evals', 'regret')].cummin()
        res[('best_func_evals', 'best_log_regret')] = res[('func_evals', 'log_regret')].cummin()
        res_l.append(res.reset_index())
    # for file in sorted(glob.glob(exp_dir + '/res/time*.csv')):
    #     tim = pd.read_csv(file, index_col=[0], header=[0])
    #     # tim = tim.groupby(by='call', axis=0).median()
    #     tim_l.append(tim.reset_index())
    # res = pd.read_csv(exp_dir+'/res/res*.csv', index_col=[0, 1], header=[0, 1])
    # tim = pd.read_csv(exp_dir+'/res/time*.csv', index_col=[0], header=[0])
    return pd.concat(res_l),os.path.basename(name)[:-5]


def load_all_exp(exp_dir_root, st, ed):
    df_res = []
    df_tim = []
    names = set()
    for directory in sorted(os.listdir(exp_dir_root))[st:ed]:
        res , name = load_csv(os.path.join(exp_dir_root, directory))
        names.update([name])
        assert len(names) < 2
        df_res.append(res)
        # df_tim.append(tim)
        # df_res.append(res)
        # df_tim.append(tim)
    # df_res = pd.concat(df_res)
    # df_tim = pd.concat(df_tim)
    return pd.concat(df_res, axis=1)


def visualize(df_res):
    # df_res.groupby(by='call', axis=1).median()
    # fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    # fig2, ax2 = plt.subplots(figsize=(5, 5), dpi=300)
    # axs = [ax, ax2]
    # f, ax = plt.subplots(figsize=(7, 7))
    # ax.set(yscale="log")
    df_res.columns = df_res.columns.rename("search alg", level=0)
    # df_median = df_res.groupby(by='call', axis=1).median()
    # df_val = df_median.loc[:, (slice(None), 'loss', 'val')]

    df_val = df_res.xs(('func_evals', 'log_regret'), level=(1, 2), axis=1).copy()
    df_test = df_res.xs(('func_evals', 'regret'), level=(1, 2), axis=1).copy()
    df_val['call'] = df_val.index.values
    df_test['call'] = df_test.index.values
    df_val = df_val.set_index('call').stack().reset_index()
    df_test = df_test.set_index('call').stack().reset_index()
    df_val = df_val.rename(columns={0: 'log_regret'})
    df_test = df_test.rename(columns={0: 'regret'})
    df = pd.merge(df_val.reset_index(), df_test.reset_index())
    df = df.drop(['index'], axis=1).set_index(['search alg',
                                               'call']).stack().reset_index()

    df = df.rename(columns={0: 'metric', 'level_2': 'metric: '})
    # df['log loss'] = np.log(df['loss'].values)

    g = sns.FacetGrid(df,
                      col="metric: ",
                      hue="search alg",
                      height=4.5,
                      sharex=False,
                      sharey=False,
                      despine=False)
    # g.map(sns.pointplot, 'call', 'metric')
    g.map(sns.lineplot, 'call', 'metric')
    # sns.lineplot(x="call", y="log loss",
    #              hue="search alg", style="loss_type",
    #              data=df)
    # plt.legend(labels=plt.gca().get_legend_handles_labels()[1][:len(np.unique(df['search alg'].values))],bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0)
    # plt.legend()
    plt.ylabel('')
    plt.suptitle('SVM Meta-Data')
    plt.tight_layout()
    plt.savefig('./out/ana_tranfer_res{}.png'.format(time.time()))
    # plt.tight_layout()
    plt.show()



    df_val = df_res.xs(('best_func_evals', 'best_log_regret'), level=(1, 2), axis=1).copy()
    df_test = df_res.xs(('best_func_evals', 'best_regret'), level=(1, 2), axis=1).copy()
    df_val['call'] = df_val.index.values
    df_test['call'] = df_test.index.values
    df_val = df_val.set_index('call').stack().reset_index()
    df_test = df_test.set_index('call').stack().reset_index()
    df_val = df_val.rename(columns={0: 'best_log_regret'})
    df_test = df_test.rename(columns={0: 'best_regret'})
    df = pd.merge(df_val.reset_index(), df_test.reset_index())
    df = df.drop(['index'], axis=1).set_index(['search alg',
                                               'call']).stack().reset_index()

    df = df.rename(columns={0: 'metric', 'level_2': 'metric:'})
    # df['log loss'] = np.log(df['loss'].values)

    # g = sns.FacetGrid(df,
    #                   # col="metric:",
    #                   hue="search alg",
    #                   height=4.5,
    #                   sharex=False,
    #                   sharey=False,
    #                   despine=False)
    # g.map(sns.lineplot, 'call', 'metric')
    df_val.call += 1
    mask = (df_val.call) % 5 == 0
    df_val[mask]
    sns.lineplot(x="call", y="best_log_regret",
                 hue="search alg", #style="loss_type",
                 data=df_val)
    # sns.lineplot(x="call", y="log loss",
    #              hue="search alg", style="loss_type",
    #              data=df)
    # plt.legend(labels=plt.gca().get_legend_handles_labels()[1][:len(np.unique(df['search alg'].values))],bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0)

    plt.ylabel('')
    plt.suptitle('SVM Meta-Data accumulate best')
    plt.tight_layout()
    plt.savefig('./out/ana_tranfer_res_best{}.png'.format(time.time()))
    # plt.tight_layout()
    plt.show()
    sns.pointplot(x="call", y="best_log_regret", hue="search alg",  # style="loss_type",
                  data=df_val[mask])
    plt.savefig('./out/ana_tranfer_res_best{}_point_plot.png'.format(time.time()))

    plt.show()


def main():
    df_res = []
    b = 50
    for i in range(6, 0, -1):
        st = -i * b
        ed = st+b
        if ed == 0:
            ed = None
        df_res.append(load_all_exp('./exp', st=st, ed=ed))
    visualize(pd.concat(df_res,
                        axis=1,
                        keys=[
                            'GP',
                            'TST-R',
                            'TAF',
                            'RGPE(mean)',
                            'TAF(RGPE)',
                            'RMoGP',
                            # 'GP'
                        ]))
    # visualize(pd.concat(df_res,
    #                     axis=1,
    #                     keys=[
    #                         'baseline',
    #                         'TAF-R',
    #                         'TST-R',
    #                         'RGPE',
    #                         'Random Search'
    #                     ]))


if __name__ == '__main__':
    main()