from numpy.core.fromnumeric import shape
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import numpy as np
import random
import psycopg2
from itertools import permutations

from queryoptimization.QueryGraph2 import Query, Relation, Query_Init, EmptyQuery, getJoinConditions
from queryoptimization.cm1_postgres_card import cm1
from math import sqrt


class Train_Join_Job(gym.Env):
    metadata = {'render.models': ['human']}

    def __init__(self, file_path=None):
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
            # "comp_cast_type2": [],
            # "company_name2": [],
            # "info_type2": [],
            # "kind_type2": [],
            # "movie_companies2": [],
            # "movie_info_idx2": [],
            # "title2": []
            "comp_cast_type2": ["id", "kind"],
            "company_name2": ["id", "name", "country_code", "imdb_id", "name_pcode_nf", "name_pcode_sf", "md5sum"],
            "info_type2": ["id", "info"],
            "kind_type2": ["id", "kind"],
            "movie_companies2": ["id", "movie_id", "company_id", "company_type_id", "note"],
            "movie_info_idx2": ["id", "movie_id", "info_type_id", "info", "note"],
            "title2": ["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id", "phonetic_code",
                       "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum"]
        }

        # self.primary = ['akaname.md5sum', 'akaname2.md5sum', 'akaname.name', 'akaname2.name', 'akaname.surname_pcode',
        #                 'akaname2.surname_pcode', 'akaname.name_pcode_cf', 'akaname2.name_pcode_cf',
        #                 'akaname.name_pcode_nf', 'akaname2.name_pcode_nf', 'akaname.person_id', 'akaname2.person_id',
        #                 'akaname.id', 'akaname2.id', 'akatitle.episode_of_id', 'akatitle2.episode_of_id',
        #                 'akatitle.kind_id', 'akatitle2.kind_id', 'akatitle.md5sum', 'akatitle2.md5sum', 'akatitle.movie_id',
        #                 'akatitle2.movie_id', 'akatitle.phonetic_code', 'akatitle2.phonetic_code', 'akatitle.title',
        #                 'akatitle2.title', 'akatitle.production_year', 'akatitle2.production_year', 'akatitle.id',
        #                 'akatitle2.id', 'castinfo.person_role_id', 'castinfo2.person_role_id', 'castinfo.movie_id',
        #                 'castinfo2.movie_id', 'castinfo.person_id', 'castinfo2.person_id', 'castinfo.role_id',
        #                 'castinfo2.role_id', 'castinfo.id', 'castinfo2.id', 'charname.imdb_id', 'charname2.imdb_id',
        #                 'charname.md5sum', 'charname2.md5sum', 'charname.name', 'charname2.name', 'charname.surname_pcode',
        #                 'charname2.surname_pcode', 'charname.name_pcode_nf', 'charname2.name_pcode_nf', 'charname.id',
        #                 'charname2.id', 'compcasttype.kind', 'compcasttype2.kind', 'compcasttype.id', 'compcasttype2.id',
        #                 'companyname.country_code', 'companyname2.country_code', 'companyname.imdb_id',
        #                 'companyname2.imdb_id', 'companyname.md5sum', 'companyname2.md5sum', 'companyname.name',
        #                 'companyname2.name', 'companyname.name_pcode_nf', 'companyname2.name_pcode_nf',
        #                 'companyname.name_pcode_sf', 'companyname2.name_pcode_sf', 'companyname.id', 'companyname2.id',
        #                 'companytype.kind', 'companytype2.kind', 'companytype.id', 'companytype2.id',
        #                 'completecast.movie_id', 'completecast2.movie_id', 'completecast.subject_id',
        #                 'completecast2.subject_id', 'completecast.id', 'completecast2.id', 'infotype.info',
        #                 'infotype2.info', 'infotype.id', 'infotype2.id', 'keyword.keyword', 'keyword2.keyword',
        #                 'keyword.phonetic_code', 'keyword2.phonetic_code', 'keyword.id', 'keyword2.id', 'kindtype.kind',
        #                 'kindtype2.kind', 'kindtype.id', 'kindtype2.id', 'linktype.link', 'linktype2.link', 'linktype.id',
        #                 'linktype2.id', 'moviecompanies.company_id', 'moviecompanies2.company_id',
        #                 'moviecompanies.company_type_id', 'moviecompanies2.company_type_id', 'moviecompanies.movie_id',
        #                 'moviecompanies2.movie_id', 'moviecompanies.id', 'moviecompanies2.id', 'movieinfo.info_type_id',
        #                 'movieinfo2.info_type_id', 'movieinfo.movie_id', 'movieinfo2.movie_id', 'movieinfo.id',
        #                 'movieinfo2.id', 'movieinfoidx.id', 'movieinfoidx2.id', 'moviekeyword.keyword_id',
        #                 'moviekeyword2.keyword_id', 'moviekeyword.movie_id', 'moviekeyword2.movie_id', 'moviekeyword.id',
        #                 'moviekeyword2.id', 'movielink.linked_movie_id', 'movielink2.linked_movie_id',
        #                 'movielink.link_type_id', 'movielink2.link_type_id', 'movielink.movie_id', 'movielink2.movie_id',
        #                 'movielink.id', 'movielink2.id', 'name.gender', 'name2.gender', 'name.imdb_id', 'name2.imdb_id',
        #                 'name.md5sum', 'name2.md5sum', 'name.name', 'name2.name', 'name.surname_pcode',
        #                 'name2.surname_pcode', 'name.name_pcode_cf', 'name2.name_pcode_cf', 'name.name_pcode_nf',
        #                 'name2.name_pcode_nf', 'name.id', 'name2.id', 'personinfo.info_type_id', 'personinfo2.info_type_id',
        #                 'personinfo.person_id', 'personinfo2.person_id', 'personinfo.id', 'personinfo2.id', 'roletype.id',
        #                 'roletype2.id', 'roletype.role', 'roletype2.role', 'title.episode_nr', 'title2.episode_nr',
        #                 'title.episode_of_id', 'title2.episode_of_id', 'title.imdb_id', 'title2.imdb_id', 'title.kind_id',
        #                 'title2.kind_id', 'title.md5sum', 'title2.md5sum', 'title.phonetic_code', 'title2.phonetic_code',
        #                 'title.season_nr', 'title2.season_nr', 'title.title', 'title2.title', 'title.production_year',
        #                 'title2.production_year', 'title.id', 'title2.id']

        self.primary = ['aka_name.id', 'aka_name.person_id', 'aka_name.name', 'aka_name.imdb_index',
                        'aka_name.name_pcode_cf', 'aka_name.name_pcode_nf', 'aka_name.surname_pcode', 'aka_name.md5sum',
                        'aka_title.id', 'aka_title.movie_id', 'aka_title.title', 'aka_title.imdb_index',
                        'aka_title.kind_id', 'aka_title.production_year', 'aka_title.phonetic_code',
                        'aka_title.episode_of_id', 'aka_title.season_nr', 'aka_title.episode_nr', 'aka_title.note',
                        'aka_title.md5sum', 'cast_info.id', 'cast_info.person_id', 'cast_info.movie_id',
                        'cast_info.person_role_id', 'cast_info.note', 'cast_info.nr_order', 'cast_info.role_id',
                        'char_name.id', 'char_name.name', 'char_name.imdb_index', 'char_name.imdb_id',
                        'char_name.name_pcode_nf', 'char_name.surname_pcode', 'char_name.md5sum', 'comp_cast_type.id',
                        'comp_cast_type.kind', 'company_name.id', 'company_name.name', 'company_name.country_code',
                        'company_name.imdb_id', 'company_name.name_pcode_nf', 'company_name.name_pcode_sf',
                        'company_name.md5sum', 'company_type.id', 'company_type.kind', 'complete_cast.id',
                        'complete_cast.movie_id', 'complete_cast.subject_id', 'complete_cast.status_id', 'info_type.id',
                        'info_type.info', 'keyword.id', 'keyword.keyword', 'keyword.phonetic_code', 'kind_type.id',
                        'kind_type.kind', 'link_type.id', 'link_type.link', 'movie_companies.id',
                        'movie_companies.movie_id', 'movie_companies.company_id', 'movie_companies.company_type_id',
                        'movie_companies.note', 'movie_info.id', 'movie_info.movie_id', 'movie_info.info_type_id',
                        'movie_info.info', 'movie_info.note', 'movie_info_idx.id', 'movie_info_idx.movie_id',
                        'movie_info_idx.info_type_id', 'movie_info_idx.info', 'movie_info_idx.note', 'movie_keyword.id',
                        'movie_keyword.movie_id', 'movie_keyword.keyword_id', 'movie_link.id', 'movie_link.movie_id',
                        'movie_link.linked_movie_id', 'movie_link.link_type_id', 'name.id', 'name.name',
                        'name.imdb_index', 'name.imdb_id', 'name.gender', 'name.name_pcode_cf', 'name.name_pcode_nf',
                        'name.surname_pcode', 'name.md5sum', 'person_info.id', 'person_info.person_id',
                        'person_info.info_type_id', 'person_info.info', 'person_info.note', 'role_type.id',
                        'role_type.role', 'title.id', 'title.title', 'title.imdb_index', 'title.kind_id',
                        'title.production_year', 'title.imdb_id', 'title.phonetic_code', 'title.episode_of_id',
                        'title.season_nr', 'title.episode_nr', 'title.series_years', 'title.md5sum',
                        'comp_cast_type2.id', 'comp_cast_type2.kind', 'company_name2.id', 'company_name2.name',
                        'company_name2.country_code', 'company_name2.imdb_id', 'company_name2.name_pcode_nf',
                        'company_name2.name_pcode_sf', 'company_name2.md5sum', 'info_type2.id', 'info_type2.info',
                        'kind_type2.id', 'kind_type2.kind', 'movie_companies2.id', 'movie_companies2.movie_id',
                        'movie_companies2.company_id', 'movie_companies2.company_type_id', 'movie_companies2.note',
                        'movie_info_idx2.id', 'movie_info_idx2.movie_id', 'movie_info_idx2.info_type_id',
                        'movie_info_idx2.info', 'movie_info_idx2.note', 'title2.id', 'title2.title',
                        'title2.imdb_index', 'title2.kind_id', 'title2.production_year', 'title2.imdb_id',
                        'title2.phonetic_code', 'title2.episode_of_id', 'title2.season_nr', 'title2.episode_nr',
                        'title2.series_years', 'title2.md5sum']

        num_of_columns = sum(len(x) for x in self.schema.values())
        num_of_relations = len(self.schema)
        print('num_of_columns : ', num_of_columns)
        print('num_of_relations : ', num_of_relations)

        ### config for postgre
        try:
            conn = psycopg2.connect(
                database='exp', user='imdb', password='ych19960128', host='127.0.0.1', port='5433')
        except:
            print("I am unable to connect to the database")
        self.cursor = conn.cursor()

        self.observation_space = spaces.Box(0, 1, shape=(
            num_of_relations * num_of_columns,), dtype=np.float32)
        self.action_space = spaces.Discrete(
            num_of_relations * (num_of_relations - 1))
        self.reward_range = [-float(10), float(0)]

        self.action_obj = []
        self.num_of_relations = num_of_relations
        self.num_of_columns = num_of_columns

    def step(self, action_num):
        action = self.action_list[action_num]
        action_num_l = action[0]
        action_num_r = action[1]
        if (type(self.action_obj[action_num_l]) is not EmptyQuery) and (
                type(self.action_obj[action_num_r]) is not EmptyQuery):
            new_action_space = []

            for subquery in self.action_obj:
                if subquery is self.action_obj[action_num_l]:
                    new_action_space.append(
                        Query(self.action_obj[action_num_l], self.action_obj[action_num_r]))
                elif subquery not in (self.action_obj[action_num_l], self.action_obj[action_num_r]):
                    new_action_space.append(subquery)
                else:
                    new_action_space.append(EmptyQuery(
                        list(np.zeros(len(self.obs[0]), dtype=int))))
            self.action_obj = new_action_space

            costs = 0
            done_counter = 0
            for subquery in self.action_obj:
                if not ((type(subquery) is Relation) or (type(subquery) is Query)):
                    done_counter += 1
        else:
            costs = 0
            done_counter = 0

        self.obs = []
        for obj in self.action_obj:
            self.obs.append(obj.mask)

        if done_counter is len(self.action_obj) - 1:

            # costs = []
            for subquery in self.action_obj:
                if (type(subquery) is Relation) or (type(subquery) is Query):
                    try:
                        costs = -1 * ((sqrt(cm1(subquery, self.cursor) - self.cost['min'])) / (
                            sqrt(self.cost['max'] - self.cost['min'])) * 10)  # sqrt SUM
                    except:
                        print("costs: " + str(cm1(subquery, self.cursor)))
                        costs = 0
                        pass
                    if costs < -10.:
                        costs = -10.
                    # print(self.render()[0])

            self.is_done = True

        # return self.obs, costs, self.is_done, {}
        return np.matrix(self.obs).flatten().tolist()[0], costs, self.is_done, {}

    def reset(self):
        sql = random.choice(self.sql_query).replace(";", "")
        self.cost = {'max': 1.e+13, 'min': 1.e+4}
        self.query = Query_Init(sql, self.schema, self.primary)
        self.is_done = False
        self.action_obj = self.query.actions
        self.action_list = list(permutations(
            range(0, len(self.query.actions)), 2))
        self.actions = list(range(0, len(self.action_list)))
        self.obs = []
        self.table_embeds={}
        self.sql_id= self.query.sql_id
        for obj in self.action_obj:
            self.obs.append(obj.mask)
            if type(obj) is not EmptyQuery:
                self.table_embeds[obj.id] = obj.embed
        return np.matrix(self.obs).flatten().tolist()[0]

    def render(self, mode='human', close=False):
        sql = []
        for q in self.action_obj:
            if type(q) is not EmptyQuery:
                sql.append(q.__str__())
            # s,_ = q.toSql(self.join_conditions,0)
            # sql.append(s)
        return sql

    def close(self):
        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getValidActions(self):
        validActions = []
        emptyRows = []
        join_conditions = getJoinConditions()

        for i in range(0, len(self.action_obj)):
            if type(self.action_obj[i]) is EmptyQuery:
                emptyRows.append(i)

        for i in range(0, len(self.action_list)):
            flag = True
            for row in emptyRows:
                if row in self.action_list[i]:
                    flag = False
                    break

            # avoid cross-joins
            if flag:
                lname = self.action_obj[self.action_list[i][0]].name
                rname = self.action_obj[self.action_list[i][1]].name
                if " AS " in lname:
                    lname_list = [lname.split(" AS ")[1]]
                else:
                    lname_list = lname.split('_')
                if " AS " in rname:
                    rname_list = [rname.split(" AS ")[1]]
                else:
                    rname_list = rname.split('_')
                qname = '_'.join(sorted(lname_list + rname_list))
                if qname not in join_conditions:
                    flag = False
            if flag:
                validActions.append(i)
        return validActions


class Train_Join_Step_Tree_Struct(Train_Join_Job):
    def __init__(self, file_path):
        super().__init__(file_path=file_path)
        self.sql=""

    def step(self, action_num):
        action = self.action_list[action_num]
        action_num_l = action[0]
        action_num_r = action[1]
        if (type(self.action_obj[action_num_l]) is not EmptyQuery) and (
                type(self.action_obj[action_num_r]) is not EmptyQuery):
            new_action_space = []

            for subquery in self.action_obj:
                if subquery is self.action_obj[action_num_l]:
                    new_action_space.append(
                        Query(self.action_obj[action_num_l], self.action_obj[action_num_r]))
                elif subquery not in (self.action_obj[action_num_l], self.action_obj[action_num_r]):
                    new_action_space.append(subquery)
                else:
                    new_action_space.append(EmptyQuery(
                        list(np.zeros(len(self.obs[0]), dtype=int))))
            self.action_obj = new_action_space

            costs = 0
            done_counter = 0
            for subquery in self.action_obj:
                if not ((type(subquery) is Relation) or (type(subquery) is Query)):
                    done_counter += 1
        else:
            costs = 0
            done_counter = 0

        self.obs = []
        for obj in self.action_obj:
            self.obs.append(obj.mask)
        sub_trees = []
        if done_counter is len(self.action_obj) - 1:

            # costs = []

            for subquery in self.action_obj:
                if (type(subquery) is Relation) or (type(subquery) is Query):
                    try:
                        costs = -1 * ((sqrt(cm1(subquery, self.cursor) - self.cost['min'])) / (
                            sqrt(self.cost['max'] - self.cost['min'])) * 10)  # sqrt SUM
                    except:
                        import pdb; pdb.set_trace()
                        print("costs: " + str(cm1(subquery, self.cursor)))
                        costs = 0
                        pass
                    if costs < -10.:
                        costs = -10.
                    self.sql=subquery.toSql(0)
                    # print(self.render()[0])
                if (type(subquery) is Query):
                    tmp_tree = subquery.to_tree_structure()
                    sub_trees.append(tmp_tree)

            self.is_done = True

        else:

            for subquery in self.action_obj:
                if (type(subquery) is Query):
                    tmp_tree = subquery.to_tree_structure()
                    sub_trees.append(tmp_tree)
            # print(len(sub_trees))
            costs = 0
        # print("Number of Join Trees: ", len(sub_trees))

        return np.matrix(self.obs).flatten().tolist()[0], costs, self.is_done, sub_trees

    def getValidActions(self):
        validActions = []
        emptyRows = []
        join_conditions = getJoinConditions()

        for i in range(0, len(self.action_obj)):
            if type(self.action_obj[i]) is EmptyQuery:
                emptyRows.append(i)

        for i in range(0, len(self.action_list)):
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






class Test_Join_Step_Tree_Struct(Train_Join_Step_Tree_Struct):
    def __init__(self, file_path):
        super().__init__(file_path=file_path)
        self.sql_query_num = 0
        self.num_test = len(self.sql_query)


    def reset(self):
        idx_sql = self.sql_query_num % self.num_test
        sql = self.sql_query[idx_sql].replace(";", "")
        self.cost = {'max': 1.e+13, 'min': 1.e+4}
        self.query = Query_Init(sql, self.schema, self.primary)
        self.is_done = False
        self.action_obj = self.query.actions
        self.action_list = list(permutations(
            range(0, len(self.query.actions)), 2))
        self.actions = list(range(0, len(self.action_list)))
        self.obs = []
        self.table_embeds={}
        self.sql_id= self.query.sql_id
        for obj in self.action_obj:
            self.obs.append(obj.mask)
            if type(obj) is not EmptyQuery:
                self.table_embeds[obj.id] = obj.embed
        self.sql_query_num += 1
        return np.matrix(self.obs).flatten().tolist()[0]
