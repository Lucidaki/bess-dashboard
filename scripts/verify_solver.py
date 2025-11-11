"""
Solver Verification Script
Tests that PuLP and CBC solver are properly installed and working
"""

import sys
import time


def test_pulp_import():
    """Test that PuLP can be imported"""
    try:
        import pulp
        print("✅ PuLP imported successfully")
        print(f"   Version: {pulp.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import PuLP: {e}")
        print("   Install with: pip install pulp")
        return False


def test_cbc_solver():
    """Test CBC solver with a simple optimization problem"""
    try:
        import pulp

        print("\n🔧 Testing CBC solver...")

        # Create a simple test problem
        # Maximize: x
        # Subject to: x <= 5
        prob = pulp.LpProblem("test_solver", pulp.LpMaximize)
        x = pulp.LpVariable("x", lowBound=0, upBound=10)

        # Objective
        prob += x

        # Constraint
        prob += x <= 5

        # Solve
        start_time = time.time()
        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        solve_time = time.time() - start_time

        if status == pulp.LpStatusOptimal:
            print(f"✅ CBC solver working")
            print(f"   Optimal solution: x = {x.varValue}")
            print(f"   Objective value: {pulp.value(prob.objective)}")
            print(f"   Solve time: {solve_time:.4f} seconds")

            # Verify solution is correct
            if abs(x.varValue - 5.0) < 0.001:
                print("✅ Solution is correct (x=5)")
                return True
            else:
                print(f"❌ Solution incorrect: expected x=5, got x={x.varValue}")
                return False
        else:
            print(f"❌ Solver failed with status: {pulp.LpStatus[status]}")
            return False

    except Exception as e:
        print(f"❌ Error testing CBC solver: {e}")
        return False


def test_solver_performance():
    """Test solver performance with a larger problem"""
    try:
        import pulp
        import numpy as np

        print("\n⚡ Testing solver performance...")

        # Create a larger optimization problem (simulating 48 periods)
        T = 48
        prob = pulp.LpProblem("test_performance", pulp.LpMaximize)

        # Decision variables
        power = [pulp.LpVariable(f"power_{t}", lowBound=-2.5, upBound=2.5) for t in range(T)]
        soc = [pulp.LpVariable(f"soc_{t}", lowBound=10, upBound=95) for t in range(T)]

        # Fake prices
        np.random.seed(42)
        prices = np.random.uniform(30, 150, T)

        # Objective: maximize revenue
        prob += pulp.lpSum([power[t] * prices[t] * 0.5 for t in range(T)])

        # Energy balance constraints
        capacity = 5.0  # MWh
        rte = 0.85
        for t in range(T - 1):
            prob += soc[t + 1] == soc[t] - (power[t] * 0.5 / capacity * 100)

        # Initial SoC
        prob += soc[0] == 50

        # Solve
        start_time = time.time()
        status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
        solve_time = time.time() - start_time

        if status == pulp.LpStatusOptimal:
            print(f"✅ Performance test passed")
            print(f"   Problem size: {T} time periods, {T*2} variables")
            print(f"   Solve time: {solve_time:.2f} seconds")

            if solve_time < 30:
                print(f"✅ Solve time within 30 second SLA")
                return True
            else:
                print(f"⚠️  Solve time exceeds 30 second SLA")
                return False
        else:
            print(f"❌ Performance test failed with status: {pulp.LpStatus[status]}")
            return False

    except Exception as e:
        print(f"❌ Error in performance test: {e}")
        return False


def main():
    """Run all solver verification tests"""
    print("=" * 60)
    print("BESS Dashboard - Solver Verification")
    print("=" * 60)

    results = {}

    # Test 1: PuLP import
    results['import'] = test_pulp_import()

    if not results['import']:
        print("\n⚠️  Cannot proceed without PuLP. Please install dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    # Test 2: CBC solver
    results['cbc'] = test_cbc_solver()

    if not results['cbc']:
        print("\n⚠️  CBC solver not working. Please ensure:")
        print("   1. CBC solver is installed")
        print("   2. CBC is in your system PATH")
        print("   3. On Windows: Download from https://github.com/coin-or/Cbc")
        print("   4. On Linux: sudo apt-get install coinor-cbc")
        print("   5. On Mac: brew install coin-or-tools/coinor/cbc")
        sys.exit(1)

    # Test 3: Performance
    results['performance'] = test_solver_performance()

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper()}: {status}")

    if all_passed:
        print("\n🎉 All tests passed! Solver is ready for use.")
        print("\nNext steps:")
        print("1. Configure BESS asset parameters in config/config_schema.yaml")
        print("2. Prepare SCADA and market price CSV files")
        print("3. Run: python src/config_loader.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
