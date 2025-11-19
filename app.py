from flask import Flask, render_template, request, render_template_string
from ZOF_CLI import ZOFSolver
import os

# Set template folder to current directory to match requested structure
app = Flask(__name__, template_folder='.')


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    history = []
    method = None

    # Defaults
    func_str = ""
    param1 = ""
    param2 = ""
    param3 = ""
    tol = "1e-6"
    max_iter = "100"

    if request.method == 'POST':
        try:
            method = request.form.get('method')
            func_str = request.form.get('function')
            tol = float(request.form.get('tolerance'))
            max_iter = int(request.form.get('iterations'))

            # Inputs based on method
            p1 = request.form.get('param1')
            p2 = request.form.get('param2')
            p3 = request.form.get('param3')  # Used for Delta or extra params

            res = None

            if method == 'bisection':
                res = ZOFSolver.bisection(func_str, float(p1), float(p2), tol, max_iter)
            elif method == 'regula_falsi':
                res = ZOFSolver.regula_falsi(func_str, float(p1), float(p2), tol, max_iter)
            elif method == 'secant':
                res = ZOFSolver.secant(func_str, float(p1), float(p2), tol, max_iter)
            elif method == 'newton':
                res = ZOFSolver.newton_raphson(func_str, float(p1), tol, max_iter)
            elif method == 'fixed_point':
                res = ZOFSolver.fixed_point(func_str, float(p1), tol, max_iter)
            elif method == 'modified_secant':
                res = ZOFSolver.modified_secant(func_str, float(p1), float(p3), tol, max_iter)

            if res:
                root_val, final_err, num_iters, hist = res
                if root_val is None:
                    error = final_err  # Logic handles error msgs in 2nd return pos if failed
                else:
                    result = {
                        'root': root_val,
                        'error': final_err,
                        'iters': num_iters
                    }
                    history = hist
        except Exception as e:
            error = f"Input Error: {str(e)}"

        # Retain form values for UX
        return render_template('index.html',
                               result=result, error=error, history=history,
                               method=method, func_str=func_str,
                               param1=p1, param2=p2, param3=p3,
                               tol=tol, max_iter=max_iter)

    return render_template('index.html')


if __name__ == '__main__':
    # Bind to 0.0.0.0 for deployment environments
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)