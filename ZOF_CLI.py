import math
import sympy as sp
import sys


class ZOFSolver:
    """
    Core solver class containing numerical methods for finding roots.
    Returns a tuple: (root, error, iterations, history_list)
    """

    @staticmethod
    def parse_function(func_str):
        """Parses a string expression into a callable sympy function."""
        x = sp.symbols('x')
        try:
            # Allow mathematical functions like sin, cos, exp, log
            expr = sp.sympify(func_str)
            return expr, x
        except Exception as e:
            return None, None

    @staticmethod
    def evaluate(expr, x_symbol, val):
        return float(expr.subs(x_symbol, val))

    @staticmethod
    def bisection(func_str, a, b, tol=1e-6, max_iter=100):
        expr, x = ZOFSolver.parse_function(func_str)
        history = []

        fa = ZOFSolver.evaluate(expr, x, a)
        fb = ZOFSolver.evaluate(expr, x, b)

        if fa * fb >= 0:
            return None, "Initial guess intervals must bracket the root (f(a) * f(b) < 0).", 0, []

        for i in range(1, max_iter + 1):
            c = (a + b) / 2
            fc = ZOFSolver.evaluate(expr, x, c)
            error = abs(b - a)

            history.append({
                'iter': i, 'a': f"{a:.6f}", 'b': f"{b:.6f}",
                'x_root': f"{c:.6f}", 'f(x)': f"{fc:.6e}", 'error': f"{error:.6e}"
            })

            if abs(fc) < tol or error < tol:
                return c, error, i, history

            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

        return c, error, max_iter, history

    @staticmethod
    def regula_falsi(func_str, a, b, tol=1e-6, max_iter=100):
        expr, x = ZOFSolver.parse_function(func_str)
        history = []

        fa = ZOFSolver.evaluate(expr, x, a)
        fb = ZOFSolver.evaluate(expr, x, b)

        if fa * fb >= 0:
            return None, "Initial guess intervals must bracket the root.", 0, []

        for i in range(1, max_iter + 1):
            # c = (a*f(b) - b*f(a)) / (f(b) - f(a))
            try:
                c = (a * fb - b * fa) / (fb - fa)
            except ZeroDivisionError:
                return None, "Division by zero encountered.", i, history

            fc = ZOFSolver.evaluate(expr, x, c)
            error = abs(fc)  # For False Position, error is often estimated by f(c) or step size

            history.append({
                'iter': i, 'a': f"{a:.6f}", 'b': f"{b:.6f}",
                'x_root': f"{c:.6f}", 'f(x)': f"{fc:.6e}", 'error': f"{error:.6e}"
            })

            if abs(fc) < tol:
                return c, error, i, history

            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        return c, error, max_iter, history

    @staticmethod
    def secant(func_str, x0, x1, tol=1e-6, max_iter=100):
        expr, x = ZOFSolver.parse_function(func_str)
        history = []

        f0 = ZOFSolver.evaluate(expr, x, x0)
        f1 = ZOFSolver.evaluate(expr, x, x1)

        for i in range(1, max_iter + 1):
            try:
                if abs(f1 - f0) < 1e-12:
                    return None, "Denominator too small (division by zero).", i, history

                x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            except ZeroDivisionError:
                return None, "Division by zero.", i, history

            f2 = ZOFSolver.evaluate(expr, x, x2)
            error = abs(x2 - x1)

            history.append({
                'iter': i, 'x_prev': f"{x1:.6f}",
                'x_root': f"{x2:.6f}", 'f(x)': f"{f2:.6e}", 'error': f"{error:.6e}"
            })

            if error < tol:
                return x2, error, i, history

            x0, f0 = x1, f1
            x1, f1 = x2, f2

        return x1, error, max_iter, history

    @staticmethod
    def newton_raphson(func_str, x0, tol=1e-6, max_iter=100):
        expr, x = ZOFSolver.parse_function(func_str)
        deriv = sp.diff(expr, x)  # Symbolic differentiation
        history = []

        xi = x0
        for i in range(1, max_iter + 1):
            fi = ZOFSolver.evaluate(expr, x, xi)
            dfi = ZOFSolver.evaluate(deriv, x, xi)

            if abs(dfi) < 1e-12:
                return None, "Derivative too close to zero.", i, history

            xi_new = xi - fi / dfi
            error = abs(xi_new - xi)

            history.append({
                'iter': i, 'x_curr': f"{xi:.6f}",
                'x_next': f"{xi_new:.6f}", 'f(x)': f"{fi:.6e}", "f'(x)": f"{dfi:.6e}", 'error': f"{error:.6e}"
            })

            if error < tol:
                return xi_new, error, i, history

            xi = xi_new

        return xi, error, max_iter, history

    @staticmethod
    def fixed_point(g_func_str, x0, tol=1e-6, max_iter=100):
        """Note: User inputs g(x) where x = g(x)"""
        expr, x = ZOFSolver.parse_function(g_func_str)
        history = []

        xi = x0
        for i in range(1, max_iter + 1):
            xi_new = ZOFSolver.evaluate(expr, x, xi)
            error = abs(xi_new - xi)

            history.append({
                'iter': i, 'x_curr': f"{xi:.6f}",
                'g(x)': f"{xi_new:.6f}", 'error': f"{error:.6e}"
            })

            if error < tol:
                return xi_new, error, i, history

            xi = xi_new

            # Divergence check
            if abs(xi) > 1e10:
                return None, "Method Diverged.", i, history

        return xi, error, max_iter, history

    @staticmethod
    def modified_secant(func_str, x0, delta, tol=1e-6, max_iter=100):
        expr, x = ZOFSolver.parse_function(func_str)
        history = []

        xi = x0
        for i in range(1, max_iter + 1):
            fi = ZOFSolver.evaluate(expr, x, xi)
            fi_delta = ZOFSolver.evaluate(expr, x, xi + delta * xi)

            denom = fi_delta - fi
            if abs(denom) < 1e-12:
                return None, "Denominator too small.", i, history

            xi_new = xi - (delta * xi * fi) / denom
            error = abs(xi_new - xi)

            history.append({
                'iter': i, 'x_curr': f"{xi:.6f}",
                'x_next': f"{xi_new:.6f}", 'f(x)': f"{fi:.6e}", 'error': f"{error:.6e}"
            })

            if error < tol:
                return xi_new, error, i, history

            xi = xi_new

        return xi, error, max_iter, history


# --- CLI Implementation ---
def run_cli():
    print("=========================================")
    print("   Zero of Functions (ZOF) Solver CLI    ")
    print("=========================================")

    while True:
        print("\nSelect Method:")
        print("1. Bisection Method")
        print("2. Regula Falsi Method")
        print("3. Secant Method")
        print("4. Newton-Raphson Method")
        print("5. Fixed Point Iteration")
        print("6. Modified Secant Method")
        print("q. Quit")

        choice = input("\nEnter choice: ").strip().lower()
        if choice == 'q':
            break

        try:
            if choice in ['1', '2']:
                eq = input("Enter f(x) (e.g., x**2 - 4): ")
                a = float(input("Enter guess a: "))
                b = float(input("Enter guess b: "))
                tol = float(input("Enter tolerance (e.g., 1e-6): "))
                iters = int(input("Enter max iterations: "))

                if choice == '1':
                    res = ZOFSolver.bisection(eq, a, b, tol, iters)
                else:
                    res = ZOFSolver.regula_falsi(eq, a, b, tol, iters)

            elif choice == '3':
                eq = input("Enter f(x): ")
                x0 = float(input("Enter x0: "))
                x1 = float(input("Enter x1: "))
                tol = float(input("Enter tolerance: "))
                iters = int(input("Enter max iterations: "))
                res = ZOFSolver.secant(eq, x0, x1, tol, iters)

            elif choice == '4':
                eq = input("Enter f(x): ")
                x0 = float(input("Enter initial guess x0: "))
                tol = float(input("Enter tolerance: "))
                iters = int(input("Enter max iterations: "))
                res = ZOFSolver.newton_raphson(eq, x0, tol, iters)

            elif choice == '5':
                print("NOTE: For Fixed Point, enter g(x) such that x = g(x)")
                eq = input("Enter g(x): ")
                x0 = float(input("Enter initial guess x0: "))
                tol = float(input("Enter tolerance: "))
                iters = int(input("Enter max iterations: "))
                res = ZOFSolver.fixed_point(eq, x0, tol, iters)

            elif choice == '6':
                eq = input("Enter f(x): ")
                x0 = float(input("Enter x0: "))
                delta = float(input("Enter perturbation delta (e.g., 0.01): "))
                tol = float(input("Enter tolerance: "))
                iters = int(input("Enter max iterations: "))
                res = ZOFSolver.modified_secant(eq, x0, delta, tol, iters)
            else:
                print("Invalid selection.")
                continue

            # Display Results
            root, err_msg_or_val, num_iter, history = res

            print("\n--- Iteration History ---")
            if history:
                headers = list(history[0].keys())
                print(" | ".join(headers))
                for step in history:
                    print(" | ".join(str(val) for val in step.values()))

            print("\n--- Final Result ---")
            if root is not None:
                print(f"Root found: {root}")
                print(f"Final Error: {err_msg_or_val}")
                print(f"Iterations: {num_iter}")
            else:
                print(f"Failed to find root: {err_msg_or_val}")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_cli()