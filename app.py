from flask import *  
import os
import demo
app = Flask(__name__)  


picFolder= os.path.join('static','pics')

app.config['UPLOAD_FOLDER']=picFolder
@app.route('/', methods = ['GET','POST'])  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['GET','POST'])  
def success():
    global count
    if request.method == 'POST':
        f = request.files['file']
        f2=request.files['file2']
        f.filename="im1.png"
        f2.filename="im2.png"
        f.save(f.filename)
        f2.save(f2.filename)
        demo.main("im1.png","im2.png")
        pic=os.path.join(app.config['UPLOAD_FOLDER'],'im_interp.png')
        #return render_template("success.html", name = f.filename, name2=f2.filename)
        return render_template("success_finished.html",imag=pic)

if __name__ == '__main__':  
    app.run()  