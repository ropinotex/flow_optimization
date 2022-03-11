# ==============================================================================
# description     :Basic flow optimization model example
# author          :Roberto Pinto
# date            :2022.03.08
# version         :1.0
# notes           :This software is meant for teaching purpose only and it is provided as-is.
#                  The model is inspired by the SlickOil case (https://slickoil.opexanalytics.com/).
#                  All data has been taken from the case. 
#                  The software is provided as-is, with no guarantee by the author.
# ==============================================================================

import pulp as pl
from itertools import product

# Data
wells_unit_production_cost = {'1': 3,
                              '2': 12,
                              '3': 4,
                              '4': 6,
                              '5': 10,
                              '6': 2.5}

wells_capacities = {'1': 40,
                    '2': 50,
                    '3': 40,
                    '4': 100,
                    '5': 100,
                    '6': 100}

refineries_unit_production_cost = {'A': 4,
                                   'B': 2,
                                   'C': 6,
                                   'D': 1,
                                   'E': 8}

refineries_capacities = {'A': 100,
                         'B': 40,
                         'C': 80,
                         'D': 100,
                         'E': 80}

allowed_routes = {('1', 'A'), ('1', 'B'), ('2', 'B'), ('2', 'C'), ('2', 'D'),
                  ('3', 'B'), ('3', 'C'), ('3', 'E'), ('4', 'A'), ('4', 'E'),
                  ('5', 'D'), ('5', 'E'), ('6', 'C'), ('6', 'E')}

demands = {'C1': 100}


def optimal_flows_node(demands,
                       wells_unit_production_cost,
                       wells_capacities,
                       refineries_unit_production_cost,
                       refineries_capacities,
                       allowed_routes,
                       blocked_wells=None,
                       blocked_refineries=None):
    
    # define the problem container
    pb = pl.LpProblem("SlickOil", pl.LpMinimize)
    
    # define sets from passed data
    wells = list(wells_unit_production_cost.keys())
    refineries = list(refineries_unit_production_cost.keys())
    customers = list(demands.keys())
    
    # define the variables representing the flow from wells (w) to refineries (r)
    flow_vars = pl.LpVariable.dicts(name="Flow",
                                    indices=[(w, r) for w in wells for r in refineries],
                                    lowBound=0,
                                    upBound=100,
                                    cat=pl.LpContinuous)
    # define the objective function (sum of all production costs)
    total_production_cost = 0
    for w in wells:
        total_production_cost += pl.lpSum([wells_unit_production_cost[w] * flow_vars[w, r] for r in refineries])

    for r in refineries:
        total_production_cost += pl.lpSum([refineries_unit_production_cost[r] * flow_vars[w, r] for w in wells])

    # setting problem objective
    pb.setObjective(total_production_cost)    
    
    # set constraints
    # capacity constraints
    for w in wells:
        pb += pl.LpConstraint(e = pl.lpSum([flow_vars[w, r] for r in refineries]), 
                              sense=pl.LpConstraintLE,
                              name=f"well_{w}_production", 
                              rhs=wells_capacities[w])
    for r in refineries:
        pb += pl.LpConstraint(e = pl.lpSum([flow_vars[w, r] for w in wells]), 
                              sense=pl.LpConstraintLE,
                              name=f"refinery_{r}_production", 
                              rhs=refineries_capacities[r])

    # demand satisfaction
    for c in customers:
        pb += pl.LpConstraint(e = pl.lpSum([flow_vars[w, r] for w in wells for r in refineries]),
                              sense=pl.LpConstraintEQ,
                              name='Demand_constraint',
                              rhs=demands[c])

    # block wells and refineries
    if blocked_wells and isinstance(blocked_wells, list):
        for w, r in product(blocked_wells, refineries):
            if (w, r) in allowed_routes:
                flow_vars[w, r].upBound = 0.
                # pb += pl.LpConstraint(e = flow_vars[w, r],
                #                         sense=pl.LpConstraintLE,
                #                         name=f"block_wells_flow_{w}_{r}", 
                #                         rhs=0)

    if blocked_refineries and isinstance(blocked_refineries, list):
        for w, r in product(wells, blocked_refineries):
            if (w, r) in allowed_routes:
                flow_vars[w, r].upBound = 0.
                # pb += pl.LpConstraint(e = flow_vars[w, r],
                #                       sense=pl.LpConstraintLE,
                #                       name=f"block_refinery_flow_{w}_{r}", 
                #                       rhs=0)

    # forbidden routes
    # set the upper bound to zero if the route is not allowed
    for w in wells:
        for r in refineries:
            if (w, r) not in allowed_routes:
                flow_vars[(w, r)].upBound = 0.

    # The problem is solved using PuLP's choice of Solver
    _solver = pl.PULP_CBC_CMD(keepFiles=False,
                              gapRel=0.00,
                              timeLimit=120, 
                              msg=True)
    pb.solve(solver=_solver)
    
    print("Optimization Status ", pl.LpStatus[pb.status] ) #print in Jupyter Notebook
    
    if pl.LpStatus[pb.status] == "Infeasible" :
        print("********* ERROR: Model not feasible, don't use results.")

    # print objective
    flows = {(w, r): flow_vars[w, r].varValue for w in wells for r in refineries if flow_vars[w, r].varValue > 0}
    
    print(f'TOTAL COST: {pl.value(pb.objective)}')
    print('Flows')
    for k, v in flows.items():
        print(f'{k[0]} --> {k[1]}  = {v}')
    print('-' * 20)
    if blocked_wells:
        print(f'BLOCKED WELLS: {blocked_wells}')
    if blocked_refineries:
        print(f'BLOCKED REFINERIES: {blocked_refineries}')        


def optimal_flows_arc(demands,
                      wells_unit_production_cost,
                      wells_capacities,
                      refineries_unit_production_cost,
                      refineries_capacities,
                      allowed_routes,
                      blocked_wells=None,
                      blocked_refineries=None):
    
    # define the problem container
    pb = pl.LpProblem("SlickOil", pl.LpMinimize)
    
    # define sets from passed data
    wells = list(wells_unit_production_cost.keys())
    refineries = list(refineries_unit_production_cost.keys())
    customers = list(demands.keys())
    
    # define the variables representing the flow from wells (w) to refineries (r)
    flow_vars = pl.LpVariable.dicts(name="Flow",
                                    indices=[(w, r) for (w, r) in allowed_routes],
                                    lowBound=0,
                                    upBound=100,
                                    cat=pl.LpContinuous)
    # define the objective function (sum of all production costs)
    total_production_cost = 0
    for w in wells:
        total_production_cost += pl.lpSum([wells_unit_production_cost[w] * flow_vars[w, r] for r in refineries if (w, r) in allowed_routes])

    for r in refineries:
        total_production_cost += pl.lpSum([refineries_unit_production_cost[r] * flow_vars[w, r] for w in wells if (w, r) in allowed_routes])

    # setting problem objective
    pb.setObjective(total_production_cost)    
    
    # set constraints
    # capacity constraints
    for w in wells:
        pb += pl.LpConstraint(e = pl.lpSum([flow_vars[w, r] for r in refineries if (w, r) in allowed_routes]), 
                              sense=pl.LpConstraintLE,
                              name=f"well_{w}_production", 
                              rhs=wells_capacities[w])
    for r in refineries:
        pb += pl.LpConstraint(e = pl.lpSum([flow_vars[w, r] for w in wells if (w, r) in allowed_routes]), 
                              sense=pl.LpConstraintLE,
                              name=f"refinery_{r}_production", 
                              rhs=refineries_capacities[r])

    # demand satisfaction
    for c in customers:
        pb += pl.LpConstraint(e = pl.lpSum([flow_vars[w, r] for w in wells for r in refineries if (w, r) in allowed_routes]),
                              sense=pl.LpConstraintEQ,
                              name='Demand_constraint',
                              rhs=demands[c])

    # block wells and refineries
    if blocked_wells and isinstance(blocked_wells, list):
        for w, r in product(blocked_wells, refineries):
            if (w, r) in allowed_routes:
                flow_vars[w, r].upBound = 0.
                # pb += pl.LpConstraint(e = flow_vars[w, r],
                #                         sense=pl.LpConstraintLE,
                #                         name=f"block_wells_flow_{w}_{r}", 
                #                         rhs=0)

    if blocked_refineries and isinstance(blocked_refineries, list):
        for w, r in product(wells, blocked_refineries):
            if (w, r) in allowed_routes:
                flow_vars[w, r].upBound = 0.
                # pb += pl.LpConstraint(e = flow_vars[w, r],
                #                       sense=pl.LpConstraintLE,
                #                       name=f"block_refinery_flow_{w}_{r}", 
                #                       rhs=0)

    # The problem is solved using PuLP's choice of Solver
    _solver = pl.PULP_CBC_CMD(keepFiles=False,
                              gapRel=0.00,
                              timeLimit=120, 
                              msg=True)
    pb.solve(solver=_solver)
    
    print("Optimization Status ", pl.LpStatus[pb.status] ) #print in Jupyter Notebook
    
    if pl.LpStatus[pb.status] == "Infeasible" :
        print("********* ERROR: Model not feasible, don't use results.")

    # print objective
    flows = {(w, r): flow_vars[w, r].varValue for w in wells for r in refineries if (w, r) in allowed_routes and flow_vars[w, r].varValue > 0}
    
    print(f'TOTAL COST: {pl.value(pb.objective)}')
    print('Flows')
    for k, v in flows.items():
        print(f'{k[0]} --> {k[1]}  = {v}')
    print('-' * 20)
    if blocked_wells:
        print(f'BLOCKED WELLS: {blocked_wells}')
    if blocked_refineries:
        print(f'BLOCKED REFINERIES: {blocked_refineries}')

