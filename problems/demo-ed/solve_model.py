#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from gurobipy import Model, GRB, quicksum


def _index_iter(obj):
    if isinstance(obj, int):
        return range(obj)
    if isinstance(obj, float):
        return range(int(obj))
    if isinstance(obj, dict):
        return list(obj.keys())
    if hasattr(obj, '__len__'):
        return range(len(obj))
    raise TypeError(f'Unsupported dimension source: {obj!r}')


def main() -> None:
    data_path = Path(__file__).with_name('data.json')
    data = json.loads(data_path.read_text(encoding='utf-8'))

    # Parameters
    WaterPerLiquid = data.get('WaterPerLiquid')
    AlcoholPerLiquid = data.get('AlcoholPerLiquid')
    WaterPerFoam = data.get('WaterPerFoam')
    AlcoholPerFoam = data.get('AlcoholPerFoam')
    AvailableWater = data.get('AvailableWater')
    AvailableAlcohol = data.get('AvailableAlcohol')
    MaxLiquidProduction = data.get('MaxLiquidProduction')
    HandsCleanedPerLiquid = data.get('HandsCleanedPerLiquid')
    HandsCleanedPerFoam = data.get('HandsCleanedPerFoam')

    model = Model('OptiAgentModel')
    model.Params.OutputFlag = 1

    # Decision variables
    numLiquid = model.addVar(vtype=GRB.INTEGER, name='numLiquid')
    numFoam = model.addVar(vtype=GRB.INTEGER, name='numFoam')

    # Constraints
    model.addConstr(WaterPerLiquid * numLiquid + WaterPerFoam * numFoam <= AvailableWater)
    model.addConstr(AlcoholPerLiquid * numLiquid + AlcoholPerFoam * numFoam <= AvailableAlcohol)
    model.addConstr(numFoam >= numLiquid + 1, name="foam_more_than_liquid")
    model.addConstr(numLiquid <= MaxLiquidProduction)
    model.addConstr(numLiquid >= 0)
    model.addConstr(numFoam >= 0)

    # Objective
    model.setObjective(30 * numLiquid + 20 * numFoam, GRB.MAXIMIZE)

    model.optimize()
    status = model.Status
    if status == GRB.OPTIMAL:
        print('Optimal objective:', model.objVal)
        solution = {var.VarName: var.X for var in model.getVars()}
        print('Solution:', json.dumps(solution, indent=2))
        output_path = Path(__file__).with_name('output_solution.json')
        output_path.write_text(json.dumps({'objective': model.objVal, 'solution': solution}, indent=2), encoding='utf-8')
    else:
        print(f'Model finished with status {status}')


if __name__ == '__main__':
    main()
