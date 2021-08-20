import time, glob, os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_csv(exp_dir):
    res_l = []
    tim_l = []
    for file in sorted(glob.glob(exp_dir + '/res/res*.csv')):
        res = pd.read_csv(file, index_col=[0, 1], header=[0, 1])
        res = res.groupby(by='call', axis=0).median()
        res_l.append(res.reset_index())
    for file in sorted(glob.glob(exp_dir + '/res/time*.csv')):
        tim = pd.read_csv(file, index_col=[0], header=[0])
        # tim = tim.groupby(by='call', axis=0).median()
        tim_l.append(tim.reset_index())
    # res = pd.read_csv(exp_dir+'/res/res*.csv', index_col=[0, 1], header=[0, 1])
    # tim = pd.read_csv(exp_dir+'/res/time*.csv', index_col=[0], header=[0])
    name = glob.glob(exp_dir + '/*.yaml')
    return pd.concat(res_l), pd.concat(tim_l), os.path.basename(name[0])[:-5]


def load_all_exp(exp_dir_root):
    df_res = {}
    df_tim = {}
    for directory in os.listdir(exp_dir_root):
        res, tim, name = load_csv(os.path.join(exp_dir_root, directory))
        df_res[name] = res
        df_tim[name] = tim
        # df_res.append(res)
        # df_tim.append(tim)
    # df_res = pd.concat(df_res)
    # df_tim = pd.concat(df_tim)
    return pd.concat(df_res.values(), axis=1, keys=df_res.keys()), \
           pd.concat(df_tim.values(), axis=1, keys=df_tim.keys())


def visualize(df_res, df_tim):
    # df_res.groupby(by='call', axis=1).median()
    # fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    # fig2, ax2 = plt.subplots(figsize=(5, 5), dpi=300)
    # axs = [ax, ax2]
    df_res.columns = df_res.columns.rename("search alg", level=0)
    # df_median = df_res.groupby(by='call', axis=1).median()
    # df_val = df_median.loc[:, (slice(None), 'loss', 'val')]

    df_val = df_res.xs(('loss', 'val'), level=(1, 2), axis=1).copy()
    df_test = df_res.xs(('loss', 'test'), level=(1, 2), axis=1).copy()
    df_val['call'] = df_val.index.values
    df_test['call'] = df_test.index.values
    df_val = df_val.set_index('call').stack().reset_index()
    df_test = df_test.set_index('call').stack().reset_index()
    df_val = df_val.rename(columns={0: 'val_loss'})
    df_test = df_test.rename(columns={0: 'test_loss'})
    df = pd.merge(df_val.reset_index(), df_test.reset_index())
    df = df.drop(['index'], axis=1).set_index(['search alg',
                                               'call']).stack().reset_index()

    df = df.rename(columns={0: 'loss', 'level_2': 'loss_type'})
    df['log loss'] = np.log(df['loss'].values)

    g = sns.FacetGrid(df,
                      col="loss_type",
                      hue="search alg",
                      height=4.5,
                      sharex=False,
                      sharey=True,
                      despine=False)
    g.map(sns.lineplot, 'call', 'log loss')
    # sns.lineplot(x="call", y="log loss",
    #              hue="search alg", style="loss_type",
    #              data=df)
    # plt.legend(labels=plt.gca().get_legend_handles_labels()[1][:len(np.unique(df['search alg'].values))],bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0)

    plt.tight_layout()
    plt.savefig('./out/ana_res{}.png'.format(time.time()))
    # plt.tight_layout()
    plt.show()

    df_ = df.groupby(['search alg', 'call',
                      'loss_type']).median().reset_index()
    tmp = pd.concat([
        df_.loc[1::2, ['loss', 'log loss']].cummin(),
        df_.loc[::2, ['loss', 'log loss']].cummin()
    ]).sort_index()
    df_.loc[:, ['loss', 'log loss']] = tmp

    g = sns.FacetGrid(df_,
                      col="loss_type",
                      hue="search alg",
                      height=4.5,
                      sharex=False,
                      sharey=True,
                      despine=False)
    g.map(sns.lineplot, 'call', 'log loss')
    # sns.lineplot(x="call", y="log loss",
    #              hue="search alg", style="loss_type",
    #              data=df_)
    plt.legend(labels=plt.gca().get_legend_handles_labels()[1]
               [:len(np.unique(df['search alg'].values))],
               bbox_to_anchor=(1.05, 0.5),
               loc='center left',
               borderaxespad=0)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0)

    plt.tight_layout()
    plt.savefig('./out/ana_res{}.png'.format(time.time()))
    # plt.tight_layout()
    plt.show()
    # # ax = experiment_main(args=args)
    # axs[0].legend(fontsize=8, loc="upper right", borderaxespad=0.0)
    # axs[1].legend(fontsize=8, loc="upper right", borderaxespad=0.0)
    # fig.show()
    # fig.savefig('../out/ana_res{}.png'.format(time.time()))
    # fig2.show()
    # fig2.savefig('../out/ana_res{}.png'.format(time.time()))


def main():
    df_res, df_tim = load_all_exp('./exp')
    visualize(df_res, df_tim)


if __name__ == '__main__':
    main()