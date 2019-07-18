import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob


class Exp_analyse:
    """
    Analyse spinup result
    """

    def __init__(self, dir_path="/home/gu/project/DRL/data/cudnn_L1_BipedalWalkerHardcore-v2*_repeat_*"):
        pd.options.mode.chained_assignment = None  # ignore warning
        self.dir_path = dir_path
        self.param = None
        self.data_list = None
        self.exp_info = None
        self.compare_name = None
        self.compare_score = None
        self._read()

    def _read(self):

        file_list = glob.glob(self.dir_path)  # get all name of files in data which follow our partten
        info = []  # extract name info e.g. seq hidden
        for name in file_list:
            l = re.findall('\d+', name)
            info.append([int(x) for x in l])

        progress_list = glob.glob(self.dir_path + "/*/")
        progress_list = [x + "progress.txt" for x in progress_list]  # get */progress.txt
        self.data_list = [pd.read_csv(x, delimiter="\t") for x in progress_list]  # read all the data from */progree.txt

        # clean info
        info = pd.DataFrame(info)
        exp_info = info[[2, 3, 4, 5, 7, 10, 11, 12]]
        exp_info.columns = ["seq", "h1", "h2", "state", "h0", "beta", "tm", "repeat"]
        exp_info.loc[:, "name"] = range(0, len(exp_info))
        exp_info["hidden"] = exp_info["h1"].apply(lambda x: str(x) + "_") + exp_info["h2"].apply(lambda x: str(x))
        self.exp_info = exp_info.drop(["h1", "h2"], axis=1)
        # return self.exp_info, self.data_list

    def compare(self, param="seq", f=900):
        self.param = param
        # if self.exp_info is None:
        #     _, _ = self.read()
        l = self.exp_info.columns  # column name
        compare_name = self.exp_info.set_index(list(l[l != "name"])).sort_index().unstack(param).fillna("n")
        compare_name[compare_name == "n"] = ""
        self.compare_name = compare_name
        score = [x.loc[f:, "AverageTestEpRet"].mean() for x in self.data_list]
        df = self.exp_info.copy()
        df["score"] = score
        df = df.drop(["name"], axis=1)
        col_name = df.columns
        self.compare_score = df.set_index(list(col_name[col_name != "score"])).sort_index().unstack(param).fillna("")
        return self.compare_name, self.compare_score

    def plot(self,
             param="hidden",
             compare=(1, 2),
             item="AverageTestEpRet"):

        if self.param is None:
            _ = self.compare(param=param)
        # d = self.compare_name

        fig, ax = plt.subplots()
        for x in compare:
            self.data_list[x][item].plot(ax=ax, figsize=(8, 5), title=item + "_" + self.param)
        ax.legend(compare)
        print(5 * "\t" + "Experiment Info")
        for c in compare:
            print("<" * 50)
            print(self.exp_info[self.exp_info.name == c])
        plt.show()


if __name__ == "__main__":
    exp_a = Exp_analyse()
    # print(exp_a.exp_info)
    compare_name, compare_score = exp_a.compare(param="seq")
    print(compare_name, compare_score)
    exp_a.plot(compare=(6, 8))
