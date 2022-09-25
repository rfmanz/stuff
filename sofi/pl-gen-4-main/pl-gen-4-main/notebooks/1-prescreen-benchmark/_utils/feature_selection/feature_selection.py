import shap, copy, sys, os, copy
import lightgbm as lgb
import numpy as np
import pandas as pd
import rdsutils
import pickle as pkl

from tqdm import tqdm
from smart_open import open
from src.Trainer import LGBMTrainer, TFTrainer
from rdsutils.feature_selection import mrmr
from rdsutils.feature_selection.WeightedCorr import WeightedCorr
from rdsutils.woe import WOE_Transform


def remove_features(candidates, to_remove, reason=None):
    result = sorted(list(set(candidates) - set(to_remove)))
    ndroped = len(candidates) - len(result)
    print(f"dropping {ndroped} features : kept {len(result)} features")
    print(f"    reason:  {reason}")
    return result


def get_feature_rankings(mrmr_features, all_features, ranking):
    ranks = list(range(1, len(mrmr_features) + 1)) + [len(all_features)] * (len(all_features) - len(mrmr_features))
    idx = mrmr_features + list(set(ranking.index) - set(mrmr_features))
    return pd.Series(ranks, idx).sort_values(ascending=True)


def get_monotone_dir(woe_dict):
    result = {}
    for k in woe_dict:
        tbl = woe_dict[k]
        if len(tbl) < 2:
            print(k, len(tbl), "no monotonic direction - probably should filter out")
            direction = 0
        elif tbl.iloc[0]["woe"] < tbl.iloc[1]["woe"]:
            direction = 1
        else:
            direction = -1
        
        result[k] = direction
    return result


def get_feature_shap_abs(shap_values, columns):
    result = pd.DataFrame(shap_values, columns=columns)\
                    .apply(lambda x: np.abs(x).mean(), axis=0)\
                    .sort_values(ascending=False)
    return result


def get_feature_by_lgbm_importance(model):
    fimp = sorted(zip(model.feature_importances_, model.feature_name_), reverse=True)
    features_by_imp = list(list(zip(*fimp))[1])    
    return features_by_imp


default_params = {
 'objective': 'binary',
 'metric': 'auc',
 'boosting': 'gbdt',
 'max_depth': 6,
 'learning_rate': 0.05,
 'min_data_in_leaf': [300],
 'verbosity': -1,
 'seed': 157,
 'n_jobs': 30,
 'n_estimators': 1000
}


class FeatureSelector(object):
    
    def __init__(self, df, model_params=default_params,
                 data_dict=None, woe_dict=None):
        """
        Two stage feature selection process for PL Gen4
        
        Because there are >5k bureau features before considering alternative options, 
        we will break feature selection into multiple stages, in a waterfall manner
        
        Developed for PL Gen4, intented for binary classification only
        """
        self.df = df
        self.params = copy.deepcopy(model_params)
        self.params_mc = copy.deepcopy(model_params)
        self.data_dict = data_dict
        # add load woe option
        

        
    def get_woe(self, df, features, target_col, weight_col):
        # woe dict
        woe = WOE_Transform(method = 'tree',min_iv = 0.01)
        woe.fit(df[features], df[target_col].astype(int),
                     Y_weight=df[weight_col], display=-1)
        df_iv = woe.get_iv()
        woe_dict = woe.woe_dict() 

        iv_tbl = df_iv[df_iv.attr.isin(features)][['attr','iv']].set_index('attr')
        iv_series = iv_tbl.iv
        return woe, woe_dict, iv_series, df_iv
    
    
    def get_lgbm_shap(self, df, features, target_col, weight_col, params):
        # shap for regular lgbm
        params = copy.deepcopy(params)
        lgbm = lgb.LGBMClassifier(**params)

        trainer = LGBMTrainer()
        trainer.train(lgbm, 
                      df,
                      features = features,
                      target_col = target_col,
                      sample_weight = df[weight_col]
                     )
        explainer = shap.TreeExplainer(lgbm)
        shap_values = explainer.shap_values(df[features])
        shap_features = get_feature_shap_abs(shap_values[1], features)

        return lgbm, shap_values, shap_features
    
    
    def get_lgbm_mc_shap(self, df, features, target_col, weight_col, params, woe_dict):
        # shap for lgbm with mc
        params = copy.deepcopy(params)
        monotone_dict = get_monotone_dir(woe_dict)
        mc = [monotone_dict[ft]
              if ft in monotone_dict else 0
              for ft in features]
        params["monotone_constraints"] = mc
        
        lgbm_mc = lgb.LGBMClassifier(**params)

        trainer_mc = LGBMTrainer()
        trainer_mc.train(lgbm_mc, 
                      df,
                      features = features,
                      target_col = target_col,
                      sample_weight = df[weight_col]
                     )
        explainer_mc = shap.TreeExplainer(lgbm_mc)
        shap_values_mc = explainer_mc.shap_values(df[features])
        shap_features_mc = get_feature_shap_abs(shap_values_mc[1], features)
        return lgbm_mc, shap_values_mc, shap_features_mc
        
    
    
    def preprocess(self, features, target_col, weight_col, output_dir=None):
        print("Preprocessing... generating iv and shaps")
        
        # woe dict
        print("prepping woe...")
        context = self.get_woe(self.df, features, target_col, weight_col)
        self.woe, self.woe_dict, self.iv_series, self.df_iv = context
        if output_dir: self.save_state_dict(output_dir)
        
        print("prepping lgbm shap")
        context = self.get_lgbm_shap(self.df, features, target_col, weight_col, self.params)
        self.lgbm, self.shap_values, self.shap_features = context
        if output_dir: self.save_state_dict(output_dir)
            
        print("prepping lgbm mc shap")
        context = self.get_lgbm_mc_shap(self.df, features, target_col, 
                                        weight_col, self.params_mc, self.woe_dict)
        self.lgbm_mc, self.shap_values_mc, self.shap_features_mc = context
        if output_dir: self.save_state_dict(output_dir)
            
        
    def filter_by_logic_expn(self, features, target_col, weight_col):
        """
        we filter out features first by business logic applied for PLGen4
        
        this function will be hard coded for that reason
        """
        # not aa able
        if self.data_dict is None:
            raise ValueError("must provide data dict")
        not_aa_able = self.data_dict[self.data_dict['adverse actionable'] == 'N'].field_name.to_list()
        features = remove_features(features, not_aa_able, "not AA")

        # get missing
        from rdsutils.feature_selection import FeatureSelector as general_purpose_fsel
        fsel = general_purpose_fsel(self.df, label_cols=target_col, feature_cols=features)
        df_missing = fsel.get_missing(0.95)
        too_many_missing = fsel.ops["missing"]
        features = remove_features(features, too_many_missing, "too many missing")
        
        # get iv
        df_iv = self.woe.get_iv()
        low_iv = df_iv[df_iv.iv<0.02].attr.to_list()
        features = remove_features(features, low_iv, "low_iv")
        return features
        
        
    def many_to_few(self, features, target_col, weight_col, 
                    nr_to_select, mrmr_denominator="mean",
                    output_dir=None):
        """
        stage 1: convert all available features to a select few
        
        options: 
            1. fit a model and look at shap
                of course it is good to do this many times...
                * lgbm
                * lgbm with mc
            2. mrmr variations
                * iv
                * shap
        """
        print("running many to few")
        
        # mrmrs
        self.ranking = pd.DataFrame(index=features)
        shap_fn = lambda X,y,w: self.shap_features.loc[features]
        shap_mc_fn = lambda X,y,w: self.shap_features_mc.loc[features]
        iv_fn = lambda X,y,w: self.iv_series.loc[features]
        
        mrmr_config = [("mrmr_shapcq_mc", shap_mc_fn), 
                       ("mrmr_shapcq", shap_fn), 
                       ("mrmr_ivcq", iv_fn)]
        for col_name, fn in mrmr_config:
            mrmr_fts = mrmr.mrmr_classif(self.df[features],
                                         self.df[target_col],
                                         relevance=fn,
                                         denominator=mrmr_denominator,
                                         K=nr_to_select)
            self.ranking[col_name] = get_feature_rankings(mrmr_fts, features, self.ranking)

        if output_dir: self.save_state_dict(output_dir)
        return self.ranking
    
    
    def fsel_shap_then_drop_corr(self, features, target_col, 
                                 weight_col, params,
                                 corr_threshold):
        """
        provided initial list of features, drop the ones with correlation > 0.8
        
        # Question: does this drop the feature with lower feature importance? 
        # What is the dropping criteria?
        """
        lgbm = lgb.LGBMClassifier(**params)
        trainer = LGBMTrainer()
        trainer.train(lgbm, 
                      self.df,
                      features=features,
                      target_col=target_col,
                      sample_weight=self.df[weight_col]
                     )
        corr_matrix = WeightedCorr(df=self.df[features+[weight_col]], 
                                   wcol=weight_col)(method='pearson')
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        features = get_feature_by_lgbm_importance(lgbm)
        fts = [f for f in features if f not in (to_drop)]
        return fts
        
    
    def fsel_on_few(self, features, target_col, 
                    weight_col, corr_threshold=0.8,
                    output_dir=None):
        """
        stage 2: with only (hopefully <= 200) hundreds of features left

        options:
            boruta
            borutaShap
            backward selection
        """
        print("running fsel on few")
        if not hasattr(self, "ranking"):
            self.ranking = pd.DataFrame(index=features)
            
        # lgbm
        fts = self.fsel_shap_then_drop_corr(features, target_col, 
                                            weight_col, self.params,
                                            corr_threshold=corr_threshold)
        self.ranking[f"lgbm_shap_{len(fts)}"] = get_feature_rankings(fts, 
                                                    self.ranking.index, self.ranking)
        
        # lgbm with mc
        params_mc = copy.deepcopy(self.params)
        monotone_dict = get_monotone_dir(self.woe_dict)
        params_mc["monotone_constraints"] = [monotone_dict[ft]
                                             if ft in monotone_dict else 0
                                             for ft in features]
        fts_mc = self.fsel_shap_then_drop_corr(features, target_col, 
                                               weight_col, params_mc,
                                               corr_threshold=corr_threshold)
        self.ranking[f"lgbm_shap_mc_{len(fts_mc)}"] = get_feature_rankings(fts_mc, 
                                                    self.ranking.index, self.ranking)
        if output_dir: self.save_state_dict(output_dir)
        return self.ranking
        
            
    def get_rankings(self, ever_selected=False):
        if ever_selected:
            return self.ranking[self.ranking.mean(axis=1) < len(self.ranking)]
        return self.ranking
    
    
    def save_state_dict(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        self.attrs = ["woe", "woe_dict", "iv_series", 
                      "lgbm", "shap_features", 
                      "lgbm_mc", "shap_features_mc"]
        for k in tqdm(self.attrs):
            if hasattr(self, k):
                path = os.path.join(dir_path, f"{k}.pkl")
                with open(path, "wb") as f:
                    pkl.dump(getattr(self, k), f)
        
        if hasattr(self, "ranking"):
            print("saving ranking.csv")
            self.ranking.to_csv(os.path.join(dir_path, "ranking.csv"))
                
    
    def load_state_dict(self, dir_path):
        files = os.listdir(dir_path)
        files = filter(lambda p: p.endswith(".pkl"), files)
        for name in files:
            attr = name.split(".")[0]
            path = os.path.join(dir_path, name)
            with open(path, "rb") as f:
                setattr(self, attr, pkl.load(f))
                
        if not hasattr(self, "ranking"):
            self.ranking = pd.read_csv(os.path.join(dir_path, "ranking.csv"), index_col=0)
    
    
    def run(self, features, target_col, weight_col, nr_to_consider, 
            nr_to_select, output_dir=None, corr_threshold=0.8,
            filter_by_logic_expn=False):
        """
        please feel free to run the module by components (actually encouraged)
        this run method is meant for a quick and dirty method to get an initial set of features 
        to play with
        
        main components are:
            preprocessing - generate info to be used/reused downstream
            many_to_few - efficient fsel methods such as mrmr, when we have too many features
            fsel_on_few - methods to be used on a small set of starting features
                - like zooming into by lgbm shap
                - boruta/boruta shap
        
        @params features: list of features to consider
        @params target_col: str - target column 
        @params weight_col: str - weight column
        @params nr_to_consider: int - nr of features to be selected in "many_to_few", such as boruta
        @params nr_to_select: int - nr of features to be selected in "fsel_on_few", 
            the idea is to eventually end up with this many features
        @params output_dir: where to store everything
            can be saved and loaded using save_state_dict and load_state_dict
        @params corr_threshold: float - threhold to filter out correlation with
        @params filter_by_logic_expn - bool
        """
        print(f"target_col: {target_col}")
        print(f"weight_col: {weight_col}")
        self.preprocess(features, target_col, weight_col, output_dir=output_dir)
        
        if filter_by_logic_expn:
            print("filtering features by logic - experian")
            features = self.filter_by_logic_expn(features, target_col, weight_col)
        
        self.many_to_few(features, target_col, weight_col, nr_to_consider)
        if output_dir: self.save_state_dict(output_dir)
         
        # get top <nr_to_select> features by mean just as a rule of a thumb
        rankings_imp = self.get_rankings(True)
        rankings_imp["<mean>"] = rankings_imp.mean(axis=1)
        rankings_imp.sort_values("<mean>", inplace=True)
        top_features = rankings_imp.index.to_list()
        rankings_imp.drop("<mean>", axis=1, inplace=True)
        
        # to approximate number of features to consider so
        # we end up nr_to_select features when using the less efficient 
        # methods
        
        approx_nr_to_select = int(nr_to_select / (corr_threshold+0.001))
        
        self.fsel_on_few(top_features[:approx_nr_to_select], target_col, 
                         weight_col, corr_threshold=corr_threshold)
        if output_dir: self.save_state_dict(output_dir)
        
        return self.get_rankings(False)
        
    
    
    