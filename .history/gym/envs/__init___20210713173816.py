from gym.envs.registration import registry, register, make, spec
# Database
# ----------------------------------------


register(
    id='CM1-postgres-card-job-v0',
    entry_point='gym.envs.database:CM1PostgresCardJob'
)

register(
    id='CM1-postgres-card-job-one-v0',
    entry_point='gym.envs.database:CM1PostgresCardJobOne'
)
register(
    id='simple-corridor-ray-v0',
    entry_point='gym.envs.database:SimpleCorridor'
)


register(
    id='CM1-postgres-card-job-cross-v0',
    entry_point='gym.envs.database:CM1PostgresCardJob0'
)
register(
    id='CM1-postgres-card-job-cross-v1',
    entry_point='gym.envs.database:CM1PostgresCardJob1'
)
register(
    id='CM1-postgres-card-job-cross-v2',
    entry_point='gym.envs.database:CM1PostgresCardJob2'
)
register(
    id='CM1-postgres-card-job-cross-v3',
    entry_point='gym.envs.database:CM1PostgresCardJob3'
)

register(
    id='PGsql_Train_Join_Job_cv0',
    kwargs={'file_path':'/data0/chenjin/join_query_optimization/agents/queries/crossval_sens/job_queries_simple_crossval_0_train.txt'},
    entry_point='gym.envs.database2:Train_Join_Job'
)


register(
    id='PGsql_Eval_Join_Job_cv0',
    kwargs={'file_path':'/data0/chenjin/join_query_optimization/agents/queries/crossval_sens/job_queries_simple_crossval_0_test.txt'},
    entry_point='gym.envs.database2:Evaluate_Join_Job',
    max_episode_steps=33
)


register(
    id='PGsql_Train_Join_Job_cv1',
    kwargs={'file_path':'/data0/chenjin/join_query_optimization/agents/queries/crossval_sens/job_queries_simple_crossval_1_train.txt'},
    entry_point='gym.envs.database2:Train_Join_Job'
)


register(
    id='PGsql_Eval_Join_Job_cv1',
    kwargs={'file_path':'/data0/chenjin/join_query_optimization/agents/queries/crossval_sens/job_queries_simple_crossval_1_test.txt'},
    entry_point='gym.envs.database2:Evaluate_Join_Job',
    max_episode_steps=33
)


register(
    id='PGsql_Train_Join_Job_cv0',
    kwargs={'file_path':'/data0/chenjin/join_query_optimization/agents/queries/crossval_sens/job_queries_simple_crossval_0_train.txt'},
    entry_point='gym.envs.database2:Train_Join_Job'
)


register(
    id='PGsql_Eval_Join_Job_cv0',
    kwargs={'file_path':'/data0/chenjin/join_query_optimization/agents/queries/crossval_sens/job_queries_simple_crossval_0_test.txt'},
    entry_point='gym.envs.database2:Evaluate_Join_Job',
    max_episode_steps=33
)

# Algorithmic
# ----------------------------------------

register(
    id='Copy-v0',
    entry_point='gym.envs.algorithmic:CopyEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='RepeatCopy-v0',
    entry_point='gym.envs.algorithmic:RepeatCopyEnv',
    max_episode_steps=200,
    reward_threshold=75.0,
)

register(
    id='ReversedAddition-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 2},
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='ReversedAddition3-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 3},
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='DuplicatedInput-v0',
    entry_point='gym.envs.algorithmic:DuplicatedInputEnv',
    max_episode_steps=200,
    reward_threshold=9.0,
)

register(
    id='Reverse-v0',
    entry_point='gym.envs.algorithmic:ReverseEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)