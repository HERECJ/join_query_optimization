from math import sqrt
# import torch
from utils import pg_utils
from env.QueryGraph import Query_Init, EmptyQuery, Query, Relation, getJoinConditions
from env.get_reward import cm3
import random
from itertools import permutations
import numpy as np


class Join_Job_Base():
    def __init__(self, file_path):
        self.sql_query = list(open(file_path))
        self.is_done = False
        self.schema = {
            "aka_name": ["id", "person_id", "name", "imdb_index", "name_pcode_cf", "name_pcode_nf", "surname_pcode",
                         "md5sum"],
            "aka_title": ["id", "movie_id", "title", "imdb_index", "kind_id", "production_year", "phonetic_code",
                          "episode_of_id", "season_nr", "episode_nr", "note", "md5sum"],
            "cast_info": ["id", "person_id", "movie_id", "person_role_id", "note", "nr_order", "role_id"],
            "char_name": ["id", "name", "imdb_index", "imdb_id", "name_pcode_nf", "surname_pcode", "md5sum"],
            "comp_cast_type": ["id", "kind"],
            "company_name": ["id", "name", "country_code", "imdb_id", "name_pcode_nf", "name_pcode_sf", "md5sum"],
            "company_type": ["id", "kind"],
            "complete_cast": ["id", "movie_id", "subject_id", "status_id"],
            "info_type": ["id", "info"],
            "keyword": ["id", "keyword", "phonetic_code"],
            "kind_type": ["id", "kind"],
            "link_type": ["id", "link"],
            "movie_companies": ["id", "movie_id", "company_id", "company_type_id", "note"],
            "movie_info": ["id", "movie_id", "info_type_id", "info", "note"],
            "movie_info_idx": ["id", "movie_id", "info_type_id", "info", "note"],
            "movie_keyword": ["id", "movie_id", "keyword_id"],
            "movie_link": ["id", "movie_id", "linked_movie_id", "link_type_id"],
            "name": ["id", "name", "imdb_index", "imdb_id", "gender", "name_pcode_cf", "name_pcode_nf", "surname_pcode",
                     "md5sum"],
            "person_info": ["id", "person_id", "info_type_id", "info", "note"],
            "role_type": ["id", "role"],
            "title": ["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id", "phonetic_code",
                      "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum"],
            "comp_cast_type2": ["id", "kind"],
            "company_name2": ["id", "name", "country_code", "imdb_id", "name_pcode_nf", "name_pcode_sf", "md5sum"],
            "info_type2": ["id", "info"],
            "kind_type2": ["id", "kind"],
            "movie_companies2": ["id", "movie_id", "company_id", "company_type_id", "note"],
            "movie_info_idx2": ["id", "movie_id", "info_type_id", "info", "note"],
            "title2": ["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id", "phonetic_code",
                       "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum"]
        }

        self.primary = [
            'aka_name.id', 'aka_name.person_id', 'aka_name.name', 'aka_name.imdb_index', 'aka_name.name_pcode_cf', 'aka_name.name_pcode_nf', 'aka_name.surname_pcode', 'aka_name.md5sum', 'aka_title.id', 'aka_title.movie_id', 'aka_title.title', 'aka_title.imdb_index', 'aka_title.kind_id', 'aka_title.production_year', 'aka_title.phonetic_code', 'aka_title.episode_of_id', 'aka_title.season_nr', 'aka_title.episode_nr', 'aka_title.note', 'aka_title.md5sum', 'cast_info.id', 'cast_info.person_id', 'cast_info.movie_id', 'cast_info.person_role_id', 'cast_info.note', 'cast_info.nr_order', 'cast_info.role_id', 'char_name.id', 'char_name.name', 'char_name.imdb_index', 'char_name.imdb_id', 'char_name.name_pcode_nf', 'char_name.surname_pcode', 'char_name.md5sum', 'comp_cast_type.id', 'comp_cast_type.kind', 'company_name.id', 'company_name.name', 'company_name.country_code', 'company_name.imdb_id', 'company_name.name_pcode_nf', 'company_name.name_pcode_sf', 'company_name.md5sum', 'company_type.id', 'company_type.kind', 'complete_cast.id', 'complete_cast.movie_id', 'complete_cast.subject_id', 'complete_cast.status_id', 'info_type.id', 'info_type.info', 'keyword.id', 'keyword.keyword', 'keyword.phonetic_code', 'kind_type.id', 'kind_type.kind', 'link_type.id', 'link_type.link', 'movie_companies.id', 'movie_companies.movie_id', 'movie_companies.company_id', 'movie_companies.company_type_id', 'movie_companies.note', 'movie_info.id', 'movie_info.movie_id', 'movie_info.info_type_id', 'movie_info.info', 'movie_info.note', 'movie_info_idx.id', 'movie_info_idx.movie_id', 'movie_info_idx.info_type_id', 'movie_info_idx.info', 'movie_info_idx.note', 'movie_keyword.id', 'movie_keyword.movie_id', 'movie_keyword.keyword_id', 'movie_link.id', 'movie_link.movie_id', 'movie_link.linked_movie_id', 'movie_link.link_type_id', 'name.id', 'name.name', 'name.imdb_index', 'name.imdb_id', 'name.gender', 'name.name_pcode_cf', 'name.name_pcode_nf', 'name.surname_pcode', 'name.md5sum', 'person_info.id', 'person_info.person_id', 'person_info.info_type_id', 'person_info.info', 'person_info.note', 'role_type.id', 'role_type.role', 'title.id', 'title.title', 'title.imdb_index', 'title.kind_id', 'title.production_year', 'title.imdb_id', 'title.phonetic_code', 'title.episode_of_id', 'title.season_nr', 'title.episode_nr', 'title.series_years', 'title.md5sum', 'comp_cast_type2.id', 'comp_cast_type2.kind', 'company_name2.id', 'company_name2.name', 'company_name2.country_code', 'company_name2.imdb_id', 'company_name2.name_pcode_nf', 'company_name2.name_pcode_sf', 'company_name2.md5sum', 'info_type2.id', 'info_type2.info', 'kind_type2.id', 'kind_type2.kind', 'movie_companies2.id', 'movie_companies2.movie_id', 'movie_companies2.company_id', 'movie_companies2.company_type_id', 'movie_companies2.note', 'movie_info_idx2.id', 'movie_info_idx2.movie_id', 'movie_info_idx2.info_type_id', 'movie_info_idx2.info', 'movie_info_idx2.note', 'title2.id', 'title2.title', 'title2.imdb_index', 'title2.kind_id', 'title2.production_year', 'title2.imdb_id', 'title2.phonetic_code', 'title2.episode_of_id', 'title2.season_nr', 'title2.episode_nr', 'title2.series_years', 'title2.md5sum'
        ]

        num_of_columns = sum(len(x) for x in self.schema.values())
        num_of_relations = len(self.schema)
        print('num_of_columns : ', num_of_columns)
        print('num_of_relations : ', num_of_relations)
        self.num_of_columns = num_of_columns
        self.num_of_relations = num_of_relations

        ### config for postgre
        try:
            conn = pg_utils.init_pg()
        except:
            print("I am unable to connect to the database")
        self.cursor = conn
    
    def step(self, action_num):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    
    def getValidActions(self):
        raise NotImplementedError

class Train_Join_Tree(Join_Job_Base):
    def __init__(self, file_path, _tree_=False):
        super().__init__(file_path)
        self._tree_ = _tree_
    
    def reset(self):
        sql = random.choice(self.sql_query).replace(";","")
        self.query = Query_Init(sql, self.schema, self.primary)

        # self.orig_sql = self.query.orig_sql
        self.is_done = False
        self.action_obj = self.query.actions
        self.action_list = list(permutations(range(0, self.num_of_relations), 2))
        # self.actions = list(range(0, len(self.action_list)))
        self.obs = []
        # self.table_embeds
        # for obj in self.action_obj:
            # self.obs.append(obj.mask)
            # if type(obj) is not EmptyQuery:
        obs_db = self.query.sql_mask
        obs_db = np.r_[obs_db, np.zeros(3*self.num_of_columns)]
        return obs_db
    
    def step(self, action_num):
        action = self.action_list[action_num]
        action_num_l = action[0]
        action_num_r = action[1]

        if (type(self.action_obj[action_num_l]) is not EmptyQuery) and (type(self.action_obj[action_num_r]) is not EmptyQuery):
            new_action_space = []
            for subquery in self.action_obj:
                if subquery is self.action_obj[action_num_l]:
                    new_action_space.append(
                        Query(self.action_obj[action_num_l], self.action_obj[action_num_r], action_num))
                elif subquery not in (self.action_obj[action_num_l], self.action_obj[action_num_r]):
                    new_action_space.append(subquery)
                else:
                    new_action_space.append(EmptyQuery(
                        np.zeros(self.num_of_columns, dtype=int)))
            self.action_obj = new_action_space

            costs = 0
            done_counter = 0
            for subquery in self.action_obj:
                if not ((type(subquery) is Relation) or (type(subquery) is Query)):
                    done_counter += 1
        else:
            costs = 0
            done_counter = 0
        
        if self._tree_:
            query_list = []
            if done_counter is len(self.action_obj) - 1:
                for subquery in self.action_obj:
                    if type(subquery) is Query:
                        query_list.append(subquery)
                        if subquery.action_num == action_num:
                            try:
                                pg_hint = "/*+Leading (" + subquery.to_hint()  + ")*/"
                                self.hint_sql =  str(pg_hint) + " " + self.query.orig_sql
                                costs = -1 *(sqrt(cm3(pg_hint, self.query.orig_sql, self.cursor)))
                            except:
                                print("costs: " + str(cm3(pg_hint, subquery, self.cursor)))
                                costs = 0
                                pass
                            self.is_done = True
            else:
                for subquery in self.action_obj:
                    if type(subquery) is Query:
                        query_list.append(subquery)
                costs = 0.
                self.is_done = False
            # raise NotImplementedError
            return None, costs, self.is_done, query_list, done_counter

        else:
            if done_counter is len(self.action_obj) - 1:
                for subquery in self.action_obj:
                    if type(subquery) is Query and subquery.action_num == action_num:
                        left_emb = subquery.left.mask
                        right_emb = subquery.right.mask
                        all_selec = self.query.sql_mask
                        all_obs = subquery.mask
                        obs_emb = np.r_[all_selec, all_obs, left_emb, right_emb]
                        try:
                            pg_hint = "/*+Leading (" + subquery.to_hint()  + ")*/"
                            self.hint_sql =  str(pg_hint) + " " + self.query.orig_sql
                            # print(self.hint_sql)
                            costs = -1 *(sqrt(cm3(pg_hint, self.query.orig_sql, self.cursor)))
                        except:
                            print("costs: " + str(cm3(pg_hint, subquery, self.cursor)))
                            costs = 0
                            pass

                        self.is_done = True
                        break
            else:
                for subquery in self.action_obj:
                    if type(subquery) is Query and subquery.action_num == action_num:
                        left_emb = subquery.left.mask
                        right_emb = subquery.right.mask
                        all_selec = self.query.sql_mask
                        all_obs = subquery.mask
                        obs_emb = np.r_[all_selec, all_obs, left_emb, right_emb]
                        break
                costs = 0.
                self.is_done = False
            return obs_emb, costs, self.is_done, [], done_counter
    
    def getValidActions(self):
        validActions = []
        emptyRows = []
        join_conditions = getJoinConditions()

        for i in range(0, len(self.action_obj)):
            if type(self.action_obj[i]) is EmptyQuery:
                emptyRows.append(i)

        for i in range(0, len(self.action_list)): #循环 （0，1） （0，2）
            flag = True
            for row in emptyRows:
                if row in self.action_list[i]:
                    flag = False
                    break

            # avoid cross-joins
            if flag:
                lname = self.action_obj[self.action_list[i][0]].name
                rname = self.action_obj[self.action_list[i][1]].name
                lname_list = lname.split('&')
                rname_list = rname.split('&')

                qname = '&'.join(sorted(lname_list + rname_list))
                if qname not in join_conditions:
                    flag = False
            if flag:
                validActions.append(i)
        return validActions

class Test_Join_Tree(Train_Join_Tree):
    def __init__(self, file_path, _tree_=False):
        super().__init__(file_path, _tree_=_tree_)
        self.sql_query_num = -1
        self.num_test = len(self.sql_query)
    

    def reset(self):
        self.sql_query_num += 1
        idx_sql = self.sql_query_num % self.num_test
        sql = self.sql_query[idx_sql].replace(";", "")
        self.query = Query_Init(sql, self.schema, self.primary)

        self.is_done = False
        self.action_obj = self.query.actions
        self.action_list = list(permutations(range(0, self.num_of_relations), 2))
        self.obs = []
        obs_db = self.query.sql_mask
        obs_db = np.r_[obs_db, np.zeros(3*self.num_of_columns)]
        return obs_db