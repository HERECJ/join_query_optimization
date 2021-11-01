from io import SEEK_CUR
import numpy as np
from numpy.lib import math
from utils import pg_utils
import  torch
import ast
import torch
join_conditions = {}
join_conditions_id = {}
column_id = {}


class Query_Init():
    def __init__(self, sqlquery, schema, indices):
        global join_conditions, join_conditions_id, column_id
        schema_one_hot = np.zeros(sum(len(col) for _, col in schema.items()), dtype=int)
        relations = {}
        relations_id = {}
        pointer = 0
        id_idx = 0
        #print(sqlquery)

        self.sql_id = sqlquery.split('|')[0]
        sql = sqlquery.split('|')[1].split('$$')[0].strip(";")

        self.orig_sql = sql.replace("IMDB", "AND")+";"
        selectivity_value = sqlquery.split('|')[1].split('$$')[1]
        selectivity_value = ast.literal_eval(selectivity_value)
        selectivity_clause = sqlquery.split('|')[1].split('$$')[2]
        selectivity_clause = ast.literal_eval(selectivity_clause)


        for val, col in schema.items():
            relations[val] = schema_one_hot.copy() 
            relations_id[val] = id_idx 

            id_idx += 1
            # relations_cum[val] = pointer
            for i in range(0, len(col)):
                column_name = val + '.' + col[i]
                column_id[column_name] = pointer  #{a.col=1,}
                relations[val][pointer] = 1
                pointer += 1
        self.actions = self.get_Actions(
            sql, relations, relations_id, schema, indices)

        join_conditions, join_conditions_id, self.link_mtx,self.sql_mask = self.get_Conditions(
            sql,  selectivity_value, selectivity_clause, self.actions, relations_id, column_id)

        
        for i in range(len(self.actions),len(schema)):
            self.actions.append(EmptyQuery(schema_one_hot))
        
        # sorted action space
        sorted_actions = []
        for key, value in schema.items():
            flag = True
            for action in self.actions:
                name = action.name
                if name == key:
                    sorted_actions.append(action)
                    flag = False
                    break
            
            if flag: sorted_actions.append(EmptyQuery(schema_one_hot))
        self.actions = sorted_actions


    def get_Actions(self, sql, masks, relations_id, schema, indices):
        action_space = []

        try:
            relations = sql.split('FROM')[1].split(
                'WHERE')[0].replace(" ", "").split(',')
        except Exception:
            relations = sql.split('FROM')[1].replace(" ", "")
            pass
        self.table_num = len(relations)



        for r in relations:
            r = r.replace("AS", " AS ")
            r_split = r.split(" AS ")

            join_table_name = r_split[-1]  # if no AS, then the final
            action_space.append(Relation(
                r, join_table_name, masks[join_table_name], schema[join_table_name], relations_id[join_table_name], indices))

        return action_space


    def get_selectivity_from_clause(self,select,clause):
        cursor = pg_utils.init_pg()
        table_name = select.split(".")[0].strip()
        if "2" in table_name:
            select_table_name = table_name.replace("2","")
        else:
            select_table_name = table_name
        print("select_table_name",select_table_name)
        sql1 = """ EXPLAIN  select * from """ +   select_table_name + " AS " + table_name + " WHERE " +  clause +";"
        print("sql1",sql1)
        cursor.execute(sql1)
        rows = cursor.fetchall()
        row0 = rows[0][0].split("(cost=")[1].split(' ')
        estimatedRows = float(row0[1].replace("rows=", ""))

        sql2 = """ EXPLAIN  select * from """ +   select_table_name +";"
        print("sql2", sql2)
        cursor.execute(sql2)
        rows = cursor.fetchall()
        row0 = rows[0][0].split("(cost=")[1].split(' ')
        all_rows = float(row0[1].replace("rows=", ""))
        print(estimatedRows)
        print(all_rows)
        return estimatedRows/all_rows





    def get_Conditions(self, sql, selectivity_value, selectivity_clause, action_space, relations_id, column_id):
        num_relations = len(relations_id)
        link_mtx = np.zeros((num_relations, num_relations))
        try:
            all_join_conditions = sql.split(' WHERE ')[1].split(' IMDB ')[1].split(' AND ')
        except Exception:
            all_join_conditions = []
            pass

        sql_mask = np.zeros(len(column_id))
        for select in selectivity_value:
           selectivity = selectivity_value[select]
           clause = selectivity_clause[select] #
           table_name = select.split('.')[0]
           for action in action_space:
               if action.name == table_name:
                #    action.set_clause(clause)
                   action.set_selectivity(select, selectivity)
               for k,v in action.embed.items():
                   sql_mask[k] = v[1]


        for condition in all_join_conditions:
            if "=" in condition:
                clauses = condition.split(' = ')
                element = []
                for clause in clauses:
                    element.append(clause.replace(
                        "\n", "").replace(" ", "").split("."))
                try:
                    l_table = element[0][0]
                    r_table = element[1][0]
                    l_col = element[0][0] + '.' + element[0][1]
                    r_col = element[1][0] + '.' + element[1][1]
                    if relations_id[l_table] >relations_id[r_table] :
                        l_table, r_table = r_table, l_table
                        l_col, r_col = r_col, l_col

                    cond_tables = '&'.join(
                        sorted([l_table] + [r_table]))
                    try:
                        l_rel_id = relations_id[l_table]
                        r_rel_id = relations_id[r_table]
                        l_id = column_id[l_col]
                        r_id = column_id[r_col]
                        temp=0
                        for action in action_space:
                            if action.id == l_rel_id:
                                action.set_join(l_id)
                                temp +=1
                            elif action.id == r_rel_id:
                                action.set_join(r_id)
                                temp += 1
                            if temp==2:
                                continue

                    except Exception:
                        pass
                    join_conditions[cond_tables] = [l_col, r_col]
                    join_conditions_id[cond_tables] = [column_id[l_col], column_id[r_col]]
                    link_mtx[l_rel_id][r_rel_id] = 1.0
                    link_mtx[r_rel_id][l_rel_id] = 1.0


                except Exception:
                    pass
        return join_conditions, join_conditions_id, link_mtx, sql_mask


class Relation(object):
    def __init__(self, name, tab_name, mask, columns, table_id, indices):
        self.name = tab_name
        self.sql_name = name
        self.mask = mask
        self.id = table_id
        # self.indices = []
        self.columns = [] #abc.id...
        # self.clauses = []
        self.selectivity={}
        self.embed={}

        for column in columns:
            column_name = tab_name + '.' + column
            self.columns.append(column_name)
            self.embed[column_id[column_name]]= [column_id[column_name],1,0,table_id]

        # for i in indices:
        #     if " AS " in self.name:
        #         table = self.name.split(" AS ")[1] #有_
        #     else:
        #         table = self.name
        #     if i.split('.')[0] == "".join(table.split("_")):
        #         self.indices.append(table+i.split('.')[1]) # 可能有问题 没有考虑_缩写的  #考虑的relation的所有列

    def set_selectivity(self,select,selectivity):
        table_name = select.split('.')[0]
        assert table_name == self.name, ValueError(
            'Does not match the table name!!!')
        """
            Save the selection for generate the sql
        """
        self.selectivity[select]=selectivity
        self.embed[column_id[select]][1] = selectivity

    def set_join(self,column_id):
        self.embed[column_id][2] = 1


    # def set_clause(self, selection): # 存select的条件
    #     table_name = selection.split(" ")[0].split('.')[0]
    #     if "(" in table_name:
    #         table_name=table_name.replace("(","")
    #     assert table_name == self.name, ValueError(
    #         'Does not match the table name!!!')
    #     """
    #         Save the selection for generate the sql
    #     """
    #     self.clauses.append(selection)
    
    # def clause_to_sql(self):
    #     if len(self.clauses) == 0:
    #         return self.sql_name
    #     else:
    #         sql = '( SELECT * FROM ' + self.sql_name + ' WHERE '
    #         clauses = ' AND '.join(self.clauses)
    #         sql += clauses
    #         sql += ') AS ' + self.name # abc, abc2
    #         #self.clauses = [] # 会影响吗
    #         return sql
    
    # def toSql(self, level):
    #     if len(self.clauses) == 0:
    #         sql = ' SELECT * FROM ' + self.sql_name
    #         return sql
    #     else:
    #         sql = ' SELECT * FROM ' + self.sql_name + ' WHERE '
    #         clauses = ' AND '.join(self.clauses)
    #         sql += clauses

    #         # self.clauses = [] # 会影响吗
    #         return sql


class EmptyQuery(object):
    name = 'EmptyQuery'
    def __init__(self, mask):
        self.mask = mask


class Query(object):
    def __init__(self, left, right, action_num):
        global join_conditions, join_conditions_id
        self.left = left
        self.right = right
        self.action_num = action_num

        self.mask = left.mask + right.mask
        lname = self.left.name
        lname_list = lname.split("&")
        
        
        rname = self.right.name
        rname_list = rname.split("&")
        self.name = '&'.join(sorted(lname_list + rname_list))
        if self.name in join_conditions:
            self.join_condition = join_conditions[self.name]
            self.join_condition_id = join_conditions_id[self.name]
        
        self.allselevtivity = np.zeros(self.mask.shape)
        
        for k,v in self.left.embed.items():
            self.allselevtivity[k]=v[1]
            

        self.embed = self.left.embed.copy()
        for k,v in self.right.embed.items():
            if k not in self.embed:
                self.embed[k]=v
                self.allselevtivity[k]=v[1]
        
        join_conditions, join_conditions_id = self.deleteJoinCondition(lname, rname, join_conditions, join_conditions_id)

        join_conditions, join_conditions_id = self.changeJoinConditions(lname, rname, self.name, join_conditions, join_conditions_id)
    
    def changeJoinConditions(self, relA, relB, relnew, join_conditions, join_conditions_id):
        conditions = {}
        conditions_id = {}
        relB = relB.split('&')
        relA = relA.split('&')
        for key, value in join_conditions.items():
            if set(relB).issubset(key.split('&')):
                new_key = '&'.join(np.unique(sorted(key.split('&')+relnew.split('&'))))
                value2 = []
                for v in value:
                    if set(relB).issubset(v.split('.')[0].split('&')):
                        value2.append('&'.join(np.unique(sorted(v.split('.')[0].split('&')+relnew.split('&'))))+ '.' +v.split('.')[1])
                    else:
                        value2.append(v)
                if new_key in conditions:
                    conditions[new_key] = conditions[new_key] + value2
                    conditions_id[new_key] = conditions_id[new_key] + join_conditions_id[key]
                else:
                    conditions[new_key] = value2
                    conditions_id[new_key] = join_conditions_id[key]
            elif set(relA).issubset(key.split('&')):
                new_key = '&'.join(np.unique(sorted(key.split('&') + relnew.split('&'))))
                value2 = []
                for v in value:
                    if set(relA).issubset(v.split('.')[0].split('&')):
                        value2.append('&'.join(np.unique(sorted(v.split('.')[0].split('&')+relnew.split('&'))))+ '.' +v.split('.')[1])
                    else:
                        value2.append(v)
                if new_key in conditions:
                    conditions[new_key] = conditions[new_key] + value2
                    conditions_id[new_key] = conditions_id[new_key] + join_conditions_id[key]
                else:
                    conditions[new_key] = value2
                    conditions_id[new_key] = join_conditions_id[key]
                
            else:
                if key in conditions:
                    conditions[key] = conditions[key] + value
                    conditions_id[new_key] = conditions_id[new_key] + join_conditions_id[key]
                else:
                    conditions[key] = value
                    conditions_id[key] = join_conditions_id[key]

        return conditions, conditions_id
    
    def deleteJoinCondition(self, relA, relB, jc, jc_id):
        conditions = dict(jc)
        conditions_id = dict(jc_id)
        try:
            join_key = '&'.join(np.unique(sorted(relA.split('&') + relB.split('&'))))
            # del conditions[join_key]
            conditions.pop(join_key)
            conditions_id.pop(join_key)
        except Exception:
            pass
        return conditions, conditions_id

    def to_hint(self):
        if type(self.left) is Relation and type(self.right) is Relation:
            return '(' + self.left.name + ' ' + self.right.name + ')'
        elif type(self.left) is Relation and type(self.right) is Query:
            return '(' + self.left.name + ' ' + self.right.to_hint() + ')'
        elif type(self.left) is Query and type(self.right) is Relation:
            return '(' + self.left.to_hint() + ' ' + self.right.name + ')'
        elif type(self.left) is Query and type(self.right) is Query:
            return '(' + self.left.to_hint() + ' ' + self.right.to_hint() + ')'
        else:
            raise ValueError("Not supported (Sub)Query!")
    
    def get_joined_column_state(self, Granularity=4):
        left_colum_emb = []
        right_colum_emb = []
        left_join_col = []
        right_join_col = []
        
        
        for i in range(0, len(self.join_condition_id), 2):
            l_ = self.join_condition_id[i]
            r_ = self.join_condition_id[i+1]
            if l_ in self.left.embed and r_ in self.right.embed:
                left_join_col.append(l_)
                right_join_col.append(r_)
                left_colum_emb.append(Query.get_selective_emb(self.left.embed[l_], Granularity))
                right_colum_emb.append(Query.get_selective_emb(self.right.embed[r_], Granularity))

            elif r_ in self.left.embed and l_ in self.right.embed:
                left_join_col.append(r_)
                right_join_col.append(l_)
                left_colum_emb.append(Query.get_selective_emb(self.left.embed[r_], Granularity))
                right_colum_emb.append(Query.get_selective_emb(self.right.embed[l_],Granularity))
            else:
                raise ValueError('Error of Join Conditions')
        
        return left_join_col, right_join_col, left_colum_emb, right_colum_emb
    
    @staticmethod
    def get_selective_emb(col_emb, Granularity):
        # join_selectivity_embed = [0 for _ in range(self.Granularity+1)]
        #     join_selectivity_embed[-1]=col[2]
        #     if col[1]==1:
        #         join_selectivity_embed[self.Granularity-1]=1
        #     elif col[1]==0:
        #         join_selectivity_embed[0] = 1
        #     else:
        #         num = math.ceil(col[1] * self.Granularity)
        #         join_selectivity_embed[num-1] = 1
        
        join_selectivity = np.zeros(Granularity + 1, dtype=np.float32)
        join_selectivity[-1] = col_emb[2]
        if col_emb[1] == 1:
            join_selectivity[Granularity - 1] = 1
        elif col_emb[1] == 0:
            join_selectivity[0] = 1
        else:
            num = math.ceil(col_emb[1] * Granularity)
            join_selectivity[num - 1] = 1
        return join_selectivity

    


def getJoinConditions():
    return join_conditions

if __name__ == '__main__':
    import random
    schema = {
        "aka_name": ["id", "person_id", "name", "imdb_index", "name_pcode_cf", "name_pcode_nf", "surname_pcode", "md5sum"],
        "aka_title": ["id", "movie_id", "title", "imdb_index", "kind_id", "production_year", "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "note", "md5sum"],
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
        "name": ["id", "name", "imdb_index", "imdb_id", "gender", "name_pcode_cf", "name_pcode_nf", "surname_pcode", "md5sum"],
        "person_info": ["id", "person_id", "info_type_id", "info", "note"],
        "role_type": ["id", "role"],
        "title": ["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id", "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum"],
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
        "title2": ["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id", "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum"]
    }
    primary = ['akaname.md5sum', 'akaname2.md5sum', 'akaname.name', 
        'akaname2.  name', 'akaname.surname_pcode', 'akaname2.surname_pcode', 'akaname.name_pcode_cf', 'akaname2.name_pcode_cf', 'akaname.name_pcode_nf', 'akaname2.name_pcode_nf', 'akaname.person_id', 'akaname2.person_id', 'akaname.id', 'akaname2.id', 'akatitle.episode_of_id', 'akatitle2.episode_of_id', 'akatitle.kind_id', 'akatitle2.kind_id', 'akatitle.md5sum', 'akatitle2.md5sum', 'akatitle.movie_id', 'akatitle2.movie_id', 'akatitle.phonetic_code', 'akatitle2.phonetic_code', 'akatitle.title', 'akatitle2.title', 'akatitle.production_year', 'akatitle2.production_year', 'akatitle.id', 'akatitle2.id', 'castinfo.person_role_id', 'castinfo2.person_role_id', 'castinfo.movie_id', 'castinfo2.movie_id', 'castinfo.person_id', 'castinfo2.person_id', 'castinfo.role_id', 'castinfo2.role_id', 'castinfo.id', 'castinfo2.id', 'charname.imdb_id', 'charname2.imdb_id', 'charname.md5sum', 'charname2.md5sum', 'charname.name', 'charname2.name', 'charname.surname_pcode', 'charname2.surname_pcode', 'charname.name_pcode_nf', 'charname2.name_pcode_nf', 'charname.id', 'charname2.id', 'compcasttype.kind', 'compcasttype2.kind', 'compcasttype.id', 'compcasttype2.id', 'companyname.country_code', 'companyname2.country_code', 'companyname.imdb_id', 'companyname2.imdb_id', 'companyname.md5sum', 'companyname2.md5sum', 'companyname.name', 'companyname2.name', 'companyname.name_pcode_nf', 'companyname2.name_pcode_nf', 'companyname.name_pcode_sf', 'companyname2.name_pcode_sf', 'companyname.id', 'companyname2.id', 'companytype.kind', 'companytype2.kind', 'companytype.id', 'companytype2.id', 'completecast.movie_id', 'completecast2.movie_id', 'completecast.subject_id', 'completecast2.subject_id', 'completecast.id', 'completecast2.id', 'infotype.info', 'infotype2.info', 'infotype.id', 'infotype2.id', 'keyword.keyword', 'keyword2.keyword', 'keyword.phonetic_code', 'keyword2.phonetic_code', 'keyword.id', 'keyword2.id', 'kindtype.kind', 'kindtype2.kind', 'kindtype.id', 'kindtype2.id', 'linktype.link', 'linktype2.link', 'linktype.id', 'linktype2.id', 'moviecompanies.company_id', 'moviecompanies2.company_id', 'moviecompanies.company_type_id', 'moviecompanies2.company_type_id', 'moviecompanies.movie_id', 'moviecompanies2.movie_id', 'moviecompanies.id', 'moviecompanies2.id', 'movieinfo.info_type_id', 'movieinfo2.info_type_id', 'movieinfo.movie_id', 'movieinfo2.movie_id', 'movieinfo.id', 'movieinfo2.id', 'movieinfoidx.id', 'movieinfoidx2.id', 'moviekeyword.keyword_id', 'moviekeyword2.keyword_id', 'moviekeyword.movie_id', 'moviekeyword2.movie_id', 'moviekeyword.id', 'moviekeyword2.id', 'movielink.linked_movie_id', 'movielink2.linked_movie_id', 'movielink.link_type_id', 'movielink2.link_type_id', 'movielink.movie_id', 'movielink2.movie_id', 'movielink.id', 'movielink2.id', 'name.gender', 'name2.gender', 'name.imdb_id', 'name2.imdb_id', 'name.md5sum', 'name2.md5sum', 'name.name', 'name2.name', 'name.surname_pcode', 'name2.surname_pcode', 'name.name_pcode_cf', 'name2.name_pcode_cf', 'name.name_pcode_nf', 'name2.name_pcode_nf', 'name.id', 'name2.id', 'personinfo.info_type_id', 'personinfo2.info_type_id', 'personinfo.person_id', 'personinfo2.person_id', 'personinfo.id', 'personinfo2.id', 'roletype.id', 'roletype2.id', 'roletype.role', 'roletype2.role', 'title.episode_nr', 'title2.episode_nr', 'title.episode_of_id', 'title2.episode_of_id', 'title.imdb_id', 'title2.imdb_id', 'title.kind_id', 'title2.kind_id', 'title.md5sum', 'title2.md5sum', 'title.phonetic_code', 'title2.phonetic_code', 'title.season_nr', 'title2.season_nr', 'title.title', 'title2.title', 'title.production_year', 'title2.production_year', 'title.id', 'title2.id']

    sql = random.choice(list(open("/data/ygy/code_list/join2/agents/queries/crossval_sens/job_queries_simple_crossval_0_train.txt")))
    query = Query_Init(sql, schema, primary)
