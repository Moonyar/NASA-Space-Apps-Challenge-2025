from flask import Flask, render_template, request, redirect, url_for
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("init.html")

@app.route("/choose", methods=["POST"])
def choose():
    option = request.form.get("option")
    # redirect to the right landing page
    if option == "1":
        return redirect(url_for("option1"))
    elif option == "2":
        return redirect(url_for("option2"))
    elif option == "3":
        return redirect(url_for("option3"))
    elif option == "4":
        return redirect(url_for("option4"))
    else:
        return redirect(url_for("index"))

@app.route("/option1")
def option1():
    return render_template("o1.html")

@app.route("/option2")
def option2():
    return render_template("o2.html")

@app.route("/option3")
def option3():
    return render_template("o3.html")

@app.route("/option4")
def option4():
    return render_template("o4.html")
