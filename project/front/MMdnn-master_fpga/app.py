#!flask/bin/python
from flask import Flask, jsonify, abort, make_response, request, url_for, send_file
import os
from string import Template
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

caffe_convert_mxnet_upload_filelist = []
caffe_convert_mxnet_download_filelist = []

pytorch_model_upload_filelist = []
tensorflow_model_upload_filelist = []
caffe_model_upload_filelist = []
mxnet_model_upload_filelist = []

pytorch_model_upload_userinfolist = []
tensorflow_model_upload_userinfolist = []
caffe_model_upload_userinfolist = []
mxnet_model_upload_userinfolist = []

#class Config(object):
#        """▒~V~R~E~M置▒~V~R~O~B▒~V~R~U▒~V~R"""
#        # sqlalchemy▒~V~R~Z~D▒~V~R~E~M置▒~V~R~O~B▒~V~R~U▒~V~R
#        # ▒~V~R~T▒~V~R▒~V~R~H▒~V~R▒~V~R~P~M▒~V~R~Zroot
#        # ▒~V~R~F▒~V~R| ~A▒~V~R~Z123
#        # ▒~V~R~U▒~V~R▒~V~R~M▒~V~R▒~V~R~S▒~V~R~Ztest1
#        SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:1234567@127.0.0.1:3306/project"
#        # 设置sqlalchemy▒~V~R~G▒~V~R▒~V~R~J▒~V~R▒~V~R~_踪▒~V~R~U▒~V~R▒~V~R~M▒~V~R▒~V~R~S
#        SQLALCHEMY_TRACE_MODIFICATIONS = False


#app.config.from_object(Config)
#db = SQLAlchemy(app) #▒~V~R~V~R~V~R~H~]▒~V~R~V~R~V~R~K▒~V~R~V~R~V~R~L~V▒~V~R~V~R~V~R~U▒~V~R~V~R~V~R▒~V~R~V~R~V~R~M▒~V~R~V~R~V~R▒~V~R~V~R~V~R~

#class Role(db.Model):
#        __tablename__ = "tbl_roles"
#        id = db.Column(db.Integer, primary_key=True)
#
#        name = db.Column(db.String(32), unique=True) # ▒~V~R~U▒~V~R▒~V~R~M▒~V~R▒~V~R~S中▒~V~R~Z~D▒~V~R~W段
#
#        users = db.relationship("User", backref="role")  # ▒~V~R~]~^▒~V~R~U▒~V~R▒~V~R~M▒~V~R▒~V~R~S中▒~V~R~Z~D▒~V~R~W段,▒~V~R~V▒~V~R便▒~V~R~_▒~V~R询


#class User(db.Model):
#        __tablename__ = "tbl_users"  # 表▒~V~R~P~M
#        id = db.Column(db.Integer, primary_key=True)  # 主▒~V~R~T▒~V~R
#
#        name = db.Column(db.String(64), unique=True) # ▒~V~R~U▒~V~R▒~V~R~M▒~V~R▒~V~R~S中▒~V~R~Z~D▒~V~R~W段
#        email = db.Column(db.String(64), unique=True)
#        password = db.Column(db.String(128))
#        role_id = db.Column(db.Integer, db.ForeignKey("tbl_roles.id"))  # ▒~V~R~V▒~V~R~T▒~V~R
#        status = db.Column(db.String(128), unique=False)



#db.drop_all()  # ▒~V~R~E▒~V~R~Y▒~V~R▒~V~R~U▒~V~R▒~V~R~M▒~V~R▒~V~R~S中▒~V~R~Z~D▒~V~R~I~@▒~V~R~\~I▒~V~R~U▒~V~R▒~V~R~M▒~V~R
#db.create_all()  # ▒~V~R~H~[建▒~V~R~U▒~V~R▒~V~R~M▒~V~R▒~V~R~S模▒~V~R~^~K类中▒~V~R~Z~D▒~V~R~I~@▒~V~R~\~I表

#role1 = Role(name="admin")  # 添▒~V~R~J| ▒~V~R~U▒~V~R▒~V~R~M▒~V~R
#role2 = Role(name="stuff")
#role3 = Role(name="wangzhao")
#db.session.add_all([role1, role2, role3])
#db.session.commit()

#user1 = User(name="a", email="a@qq.com", password="abc", role_id=role1.id, status="begin to compute flops...")
#user2 = User(name="b", email="b@qq.com", password="abc", role_id=role2.id, status="computing flops...")
#user3 = User(name="c", email="c@qq.com", password="abc", role_id=role2.id, status="flops compute complete!!!")
#user4 = User(name="d", email="d@qq.com", password="abc", role_id=role1.id, status="flops compute complete!!!")
#user5 = User(name="e", email="e@qq.com", password="abc", role_id=role3.id, status="flops compute complete!!!")
#user6 = User(name="f", email="f@qq.com", password="abc", role_id=role3.id, status="flops compute complete!!!")
#user7 = User(name="g", email="g@qq.com", password="abc", role_id=role2.id, status="flops compute complete!!!")
#db.session.add_all([user1, user2, user3, user4, user5, user6, user7
#db.session.commit()



pwd = os.path.dirname(__file__)

UPLOAD_FOLDER = os.path.join(pwd,'save_file')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'json', 'params', 'onnx', 'prototxt', 'caffemodel', 'pth', 'ckpt', 'meta', 'pb', 'py'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HOST = "10.168.103.104"
PORT = 5003
#PORT1 = 5005

@app.route('/compiler/v1.0/HQCompiler_FPGA_Optimization', methods=['GET', 'POST'])
def HQ_model_convert():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Pytorch模型转换IR
          </div>
          <div>
          <button onclick="pytorch_to_tensorflow()">pytorch模型转换</button>
          <!--button onclick="pytorch_to_caffe()">pytorch转caffe</button>
          <button onclick="pytorch_to_mxnet()">pytorch转mxnet</button>
          <button onclick="pytorch_to_onnx()">pytorch转onnx</button-->
          </div>
          <br>
          <div>
          TensorFlow模型转换IR
          </div>
          <div>
          <button onclick="tensorflow_to_pytorch()">tensorflow模型转换</button>
          <!--button onclick="tensorflow_to_caffe()">tensorflow转caffe</button>
          <button onclick="tensorflow_to_mxnet()">tensorflow转mxnet</button>
          <button onclick="tensorflow_to_onnx()">tensorflow转onnx</button-->
          </div>
          <br>
          <div>
          Caffe模型转换IR
          </div>
          <div>
          <button onclick="caffe_to_pytorch()">caffe模型转换</button>
          <!--button onclick="caffe_to_tensorflow()">caffe转tensorflow</button>
          <button onclick="caffe_to_mxnet()">caffe转mxnet</button>
          <button onclick="caffe_to_onnx()">caffe转onnx</button-->
          </div>
          <br>
          <div>
          Mxnet模型转换IR
          </div>
          <div>
          <button onclick="mxnet_to_pytorch()">mxnet模型转换</button>
          <!--button onclick="mxnet_to_tensorflow()">mxnet转tensorflow</button>
          <button onclick="mxnet_to_caffe()">mxnet转caffe</button>
          <button onclick="mxnet_to_onnx()">mxnet转onnx</button-->
          </div>
          <br>
          <div>
          ONNX模型转换IR
          </div>
          <div>
          <button onclick="onnx_to_ir()">ONNX模型转换IR</button>
          <!--button onclick="mxnet_to_tensorflow()">mxnet转tensorflow</button>
          <button onclick="mxnet_to_caffe()">mxnet转caffe</button>
          <button onclick="mxnet_to_onnx()">mxnet转onnx</button-->
          </div>
          <script>
            function pytorch_to_tensorflow() {
                window.location.href='http://$HOST:$PORT/pytorch_to_tensorflow_model_upload'
            }
            function pytorch_to_caffe() {
                window.location.href='http://$HOST:$PORT/pytorch_to_caffe_model_upload'
            }
            function pytorch_to_mxnet() {
                window.location.href='http://$HOST:$PORT/pytorch_to_mxnet_model_upload'
            }
            function pytorch_to_onnx() {
                window.location.href='http://$HOST:$PORT/pytorch_to_onnx_model_upload'
            }
            function tensorflow_to_pytorch() {
                window.location.href='http://$HOST:$PORT/tensorflow_to_pytorch_model_upload'
            }
            function tensorflow_to_caffe() {
                window.location.href='http://$HOST:$PORT/tensorflow_to_caffe_model_upload'
            }
            function tensorflow_to_mxnet() {
                window.location.href='http://$HOST:$PORT/tensorflow_to_mxnet_model_upload'
            }
            function tensorflow_to_onnx() {
                window.location.href='http://$HOST:$PORT/tensorflow_to_onnx_model_upload'
            }
            function caffe_to_pytorch() {
                window.location.href='http://$HOST:$PORT/caffe_to_pytorch_model_upload'
            }
            function caffe_to_tensorflow() {
                window.location.href='http://$HOST:$PORT/caffe_to_tensorflow_model_upload'
            }
            function caffe_to_mxnet() {
                window.location.href='http://$HOST:$PORT/caffe_to_mxnet_model_upload'
            }
            function caffe_to_onnx() {
                window.location.href='http://$HOST:$PORT/caffe_to_onnx_model_upload'
            }
            function mxnet_to_pytorch() {
                window.location.href='http://$HOST:$PORT/mxnet_to_pytorch_model_upload'
            }
            function mxnet_to_tensorflow() {
                window.location.href='http://$HOST:$PORT/mxnet_to_tensorflow_model_upload'
            }
            function mxnet_to_caffe() {
                window.location.href='http://$HOST:$PORT/mxnet_to_caffe_model_upload'
            }
            function mxnet_to_onnx() {
                window.location.href='http://$HOST:$PORT/mxnet_to_onnx_model_upload'
            }
            function onnx_to_ir() {
                window.location.href='http://$HOST:$PORT/onnx_to_ir_model_upload'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    return html

@app.route('/pytorch_to_tensorflow_model_upload')
def pytorch_to_tensorflow_model_upload():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          pytorch模型文件上传
          <br>
          <form action = "http://$HOST:$PORT/compiler/v1.0/pytorch_model_conversion_file_upload" method = "POST"
             enctype = "multipart/form-data">
             <input type = "file" name = "file" />
             <p>input shape<input type="text" name="input_shape">
             <p>dstNode name<input type="text" name="dstNode_name">
             <input type = "submit"/>
          </form>

       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    return html

@app.route('/tensorflow_to_pytorch_model_upload')
def tensorflow_to_pytorch_model_upload():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          tensorflow模型文件上传
          <br>
          <form action = "http://$HOST:$PORT/compiler/v1.0/tensorflow_model_conversion_file_upload" method = "POST"
             enctype = "multipart/form-data">
             <p>ckpt.meta文件上传：<input type = "file" name = "file" />
             <p>ckpt文件上传：<input type = "file" name = "file" />
             <p>input shape<input type="text" name="input_shape">
             <p>dstNode name<input type="text" name="dstNode_name">
             <input type = "submit"/>
          </form>

       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    return html

@app.route('/caffe_to_pytorch_model_upload')
def caffe_to_pytorch_model_upload():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          caffe模型文件上传
          <br>
          <form action = "http://$HOST:$PORT/compiler/v1.0/caffe_model_conversion_file_upload" method = "POST"
             enctype = "multipart/form-data">
             <p>prototxt文件上传：<input type = "file" name = "file" />
             <p>caffemodel文件上传：<input type = "file" name = "file" />
             <p>input shape<input type="text" name="input_shape">
             <p>dstNode name<input type="text" name="dstNode_name">
             <input type = "submit"/>
          </form>

       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    return html

@app.route('/mxnet_to_pytorch_model_upload')
def mxnet_to_pytorch_model_upload():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          mxnet模型文件上传
          <br>
          <form action = "http://$HOST:$PORT/compiler/v1.0/mxnet_model_conversion_file_upload" method = "POST"
             enctype = "multipart/form-data">
             <p>JSON文件上传： <input type = "file" name = "file" />
             <p>Params文件上传：<input type = "file" name = "file" />
             <p>input shape<input type="text" name="input_shape">
             <p>dstNode name<input type="text" name="dstNode_name">
             <input type = "submit"/>
          </form>

       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    return html

@app.route('/onnx_to_ir_model_upload')
def onnx_to_ir_model_upload():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          ONNX
          <br>
          <form action = "http://$HOST:$PORT/compiler/v1.0/onnx_model_conversion_file_upload" method = "POST"
             enctype = "multipart/form-data">
             <p>ONNX模型文件上传： <input type = "file" name = "file" />
             <p>input shape<input type="text" name="input_shape">
             <input type = "submit"/>
          </form>

       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    return html


@app.route('/compiler/v1.0/pytorch_model_conversion_file_upload', methods=['GET', 'POST'])
def pytorchupload_file():
    pytorch_model_upload_filelist.clear()
    pytorch_model_upload_userinfolist.clear()
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Pytorch模型上传成功
          </div>
          <div>
          <button onclick="begin_convert_to_tensorflow()">开始转换IR</button>
          </div>
          <!--div>
          <button onclick="begin_convert_to_caffe()">Begin Convert to caffe</button>
          </div>
          <div>
          <button onclick="begin_convert_to_mxnet()">Begin Convert to mxnet</button>
          </div>
          <div>
          <button onclick="begin_convert_to_onnx()">Begin Convert to onnx</button>
          </div-->
          <script>
            function begin_convert_to_tensorflow() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/pytorchmodel_convert_to_tensorflow';
            }
            function begin_convert_to_caffe() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/pytorchmodel_convert_to_caffe';
            }
            function begin_convert_to_mxnet() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/pytorchmodel_convert_to_mxnet';
            }
            function begin_convert_to_onnx() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/pytorchmodel_convert_to_onnx';
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    user_info = request.values.to_dict()
    input_shape = user_info.get("input_shape")
    dstNode_name = user_info.get("dstNode_name")
    pytorch_model_upload_userinfolist.append(input_shape)
    pytorch_model_upload_userinfolist.append(dstNode_name)
    print('input_shape: '+input_shape)
    print('dstNode_name: '+dstNode_name)
    """
    ▒~V~R~J▒~V~R| ▒~V~R~V~G件▒~V~R~H▒~V~Rsave_file▒~V~R~V~G件夹
    以requests▒~V~R~J▒~V~R| 举▒~V~R~K
    wiht open('路▒~V~R~D','rb') as file_obj:
        rsp = requests.post('http://localhost:5000/upload,files={'file':file_obj})
        print(rsp.text) --> file uploaded successfully
    """
    for file in request.files.getlist('file'):
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pytorch_model_upload_filelist.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename+" upload successfully")
    #return 'files uploaded successfully'
    return html

@app.route('/compiler/v1.0/tensorflow_model_conversion_file_upload', methods=['GET', 'POST'])
def tensorflowupload_file():
    tensorflow_model_upload_filelist.clear()
    tensorflow_model_upload_userinfolist.clear()
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          tensorflow模型上传成功
          </div>
          <div>
          <button onclick="begin_convert_to_pytorch()">开始转换IR</button>
          </div>
          <!--div>
          <button onclick="begin_convert_to_caffe()">Begin Convert to caffe</button>
          </div>
          <div>
          <button onclick="begin_convert_to_mxnet()">Begin Convert to mxnet</button>
          </div>
          <div>
          <button onclick="begin_convert_to_onnx()">Begin Convert to onnx</button>
          </div-->
          <script>
            function begin_convert_to_pytorch() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/tensorflowmodel_convert_to_pytorch';
            }
            function begin_convert_to_caffe() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/tensorflowmodel_convert_to_caffe';
            }
            function begin_convert_to_mxnet() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/tensorflowmodel_convert_to_mxnet';
            }
            function begin_convert_to_onnx() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/tensorflowmodel_convert_to_onnx';
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    user_info = request.values.to_dict()
    input_shape = user_info.get("input_shape")
    dstNode_name = user_info.get("dstNode_name")
    tensorflow_model_upload_userinfolist.append(input_shape)
    tensorflow_model_upload_userinfolist.append(dstNode_name)
    print('input_shape: '+input_shape)
    print('dstNode_name: '+dstNode_name)
    """
    ▒~V~R~J▒~V~R| ▒~V~R~V~G件▒~V~R~H▒~V~Rsave_file▒~V~R~V~G件夹
    以requests▒~V~R~J▒~V~R| 举▒~V~R~K
    wiht open('路▒~V~R~D','rb') as file_obj:
        rsp = requests.post('http://localhost:5000/upload,files={'file':file_obj})
        print(rsp.text) --> file uploaded successfully
    """
    for file in request.files.getlist('file'):
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            tensorflow_model_upload_filelist.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename+" upload successfully")
    #return 'files uploaded successfully'
    return html

@app.route('/compiler/v1.0/caffe_model_conversion_file_upload', methods=['GET', 'POST'])
def caffeupload_file():
    caffe_model_upload_filelist.clear()
    caffe_model_upload_userinfolist.clear()
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Caffe模型上传成功
          </div>
          <div>
          <button onclick="begin_convert_to_pytorch()">开始转换IR</button>
          </div>
          <!--div>
          <button onclick="begin_convert_to_tensorflow()">Begin Convert to tensorflow</button>
          </div>
          <div>
          <button onclick="begin_convert_to_mxnet()">Begin Convert to mxnet</button>
          </div>
          <div>
          <button onclick="begin_convert_to_onnx()">Begin Convert to onnx</button>
          </div-->
          <script>
            function begin_convert_to_pytorch() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/caffemodel_convert_to_pytorch';
            }
            function begin_convert_to_tensorflow() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/caffemodel_convert_to_tensorflow';
            }
            function begin_convert_to_mxnet() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/caffemodel_convert_to_mxnet';
            }
            function begin_convert_to_onnx() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/caffemodel_convert_to_onnx';
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    user_info = request.values.to_dict()
    input_shape = user_info.get("input_shape")
    dstNode_name = user_info.get("dstNode_name")
    caffe_model_upload_userinfolist.append(input_shape)
    caffe_model_upload_userinfolist.append(dstNode_name)
    print('input_shape: '+input_shape)
    print('dstNode_name: '+dstNode_name)
    """
    ▒~V~R~J▒~V~R| ▒~V~R~V~G件▒~V~R~H▒~V~Rsave_file▒~V~R~V~G件夹
    以requests▒~V~R~J▒~V~R| 举▒~V~R~K
    wiht open('路▒~V~R~D','rb') as file_obj:
        rsp = requests.post('http://localhost:5000/upload,files={'file':file_obj})
        print(rsp.text) --> file uploaded successfully
    """
    for file in request.files.getlist('file'):
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            caffe_model_upload_filelist.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename+" upload successfully")
    #return 'files uploaded successfully'
    return html

@app.route('/compiler/v1.0/mxnet_model_conversion_file_upload', methods=['GET', 'POST'])
def mxnetupload_file():
    mxnet_model_upload_filelist.clear()
    mxnet_model_upload_userinfolist.clear()
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Mxnet模型上传成功
          </div>
          <div>
          <button onclick="begin_convert_to_pytorch()">开始转换IR</button>
          </div>
          <!--div>
          <button onclick="begin_convert_to_tensorflow()">Begin Convert to tensorflow</button>
          </div>
          <div>
          <button onclick="begin_convert_to_caffe()">Begin Convert to caffe</button>
          </div>
          <div>
          <button onclick="begin_convert_to_onnx()">Begin Convert to onnx</button>
          </div-->
          <script>
            function begin_convert_to_pytorch() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/mxnetmodel_convert_to_pytorch';
            }
            function begin_convert_to_tensorflow() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/mxnetmodel_convert_to_tensorflow';
            }
            function begin_convert_to_caffe() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/mxnetmodel_convert_to_caffe';
            }
            function begin_convert_to_onnx() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/mxnetmodel_convert_to_onnx';
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    user_info = request.values.to_dict()
    input_shape = user_info.get("input_shape")
    dstNode_name = user_info.get("dstNode_name")
    mxnet_model_upload_userinfolist.append(input_shape)
    mxnet_model_upload_userinfolist.append(dstNode_name)
    print('input_shape: '+input_shape)
    print('dstNode_name: '+dstNode_name)
    """
    ▒~V~R~J▒~V~R| ▒~V~R~V~G件▒~V~R~H▒~V~Rsave_file▒~V~R~V~G件夹
    以requests▒~V~R~J▒~V~R| 举▒~V~R~K
    wiht open('路▒~V~R~D','rb') as file_obj:
        rsp = requests.post('http://localhost:5000/upload,files={'file':file_obj})
        print(rsp.text) --> file uploaded successfully
    """
    for file in request.files.getlist('file'):
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            mxnet_model_upload_filelist.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename+" upload successfully")
    #return 'files uploaded successfully'
    return html

@app.route('/compiler/v1.0/onnx_model_conversion_file_upload', methods=['GET', 'POST'])
def onnxupload_file():
    pytorch_model_upload_filelist.clear()
    pytorch_model_upload_userinfolist.clear()
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          ONNX模型上传成功
          </div>
          <div>
          <button onclick="begin_convert_to_ir()">开始转换IR</button>
          </div>
          <!--div>
          <button onclick="begin_convert_to_caffe()">Begin Convert to caffe</button>
          </div>
          <div>
          <button onclick="begin_convert_to_mxnet()">Begin Convert to mxnet</button>
          </div>
          <div>
          <button onclick="begin_convert_to_onnx()">Begin Convert to onnx</button>
          </div-->
          <script>
            function begin_convert_to_ir() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/onnxmodel_convert_to_ir';
            }
            function begin_convert_to_caffe() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/pytorchmodel_convert_to_caffe';
            }
            function begin_convert_to_mxnet() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/pytorchmodel_convert_to_mxnet';
            }
            function begin_convert_to_onnx() {
                window.location.href='http://$HOST:$PORT/compiler/v1.0/pytorchmodel_convert_to_onnx';
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    user_info = request.values.to_dict()
    input_shape = user_info.get("input_shape")
    pytorch_model_upload_userinfolist.append(input_shape)
    print('input_shape: '+input_shape)
    """
    ▒~V~R~J▒~V~R| ▒~V~R~V~G件▒~V~R~H▒~V~Rsave_file▒~V~R~V~G件夹
    以requests▒~V~R~J▒~V~R| 举▒~V~R~K
    wiht open('路▒~V~R~D','rb') as file_obj:
        rsp = requests.post('http://localhost:5000/upload,files={'file':file_obj})
        print(rsp.text) --> file uploaded successfully
    """
    for file in request.files.getlist('file'):
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pytorch_model_upload_filelist.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename+" upload successfully")
    #return 'files uploaded successfully'
    return html


@app.route('/compiler/v1.0/pytorchmodel_convert_to_tensorflow', methods=['GET', 'POST'])
def pytorchmodel_convert_to_tensorflow():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Pytorch模型转IR成功
          </div>
          <div>
          <button onclick="begin_download_params()">JSON文件下载</button>
          <button onclick="begin_download_json()">pb文件下载</button>
          <button onclick="begin_download_npy()">npy文件下载</button>
          <button onclick="begin_optimization()">编译优化</button>
          <button onclick="no_optimization()">未编译优化</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=converted.json'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=converted.pb'
            }
            function begin_download_npy() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=converted.npy'
            }
            function begin_optimization() {
                window.location.href='http://$HOST:5005/compiler/v1.0/ir_fpga_optimization'
            }
            function no_optimization() {
                window.location.href='http://$HOST:5005/compiler/v1.0/ir_fpga_no_optimization'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(pytorch_model_upload_userinfolist[0])
    print(pytorch_model_upload_userinfolist[1])
    print(pytorch_model_upload_filelist[0])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    #rm_command = 'rm -r converted_model.ckpt'
    #print(rm_command)
    #mmtoir_command = 'rm -f pytorch_to_tensorflow.tar.gz && rm -rf pytorch_convert_tensorflow.ckpt && mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n ./save_file/'+pytorch_model_upload_filelist[0]
    #print(mmtoir_command)
    #val1 = os.system(mmtoir_command)
    #mmtocode_command = 'mmtocode -f tensorflow -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'tensorflow_'+pytorch_model_upload_filelist[0]+'.py'
    #print(mmtocode_command)
    #val2 = os.system(mmtocode_command)
    #mmtomodel_command = 'mmtomodel -f tensorflow -in tensorflow_'+pytorch_model_upload_filelist[0]+'.py -iw '+pytorch_model_upload_userinfolist[1]+'.npy -o pytorch_convert_tensorflow.ckpt && tar -zcvf pytorch_to_tensorflow.tar.gz ./pytorch_convert_tensorflow.ckpt'
    #print(mmtomodel_command)
    #val3 = os.system(mmtomodel_command)
    mmtoir_command = 'python3 -m mmdnn.conversion._script.convertToIR -f pytorch -n ./models/torch/model.pth -w ./models/torch/model.pth --inputShape 1,800,800 -o converted'
    print(mmtoir_command)
    val = os.system(mmtoir_command)
    print(val)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/pytorchmodel_convert_to_caffe', methods=['GET', 'POST'])
def pytorchmodel_convert_to_caffe():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Pytorch模型转caffe模型成功
          </div>
          <div>
          <button onclick="begin_download_params()">caffemodel文件下载</button>
          <button onclick="begin_download_json()">prototxt文件下载</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=converted_caffe_model.caffemodel'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=converted_caffe_model.prototxt'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(pytorch_model_upload_userinfolist[0])
    print(pytorch_model_upload_userinfolist[1])
    print(pytorch_model_upload_filelist[0])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n ./save_file/'+pytorch_model_upload_filelist[0]
    print(mmtoir_command)
    val1 = os.system(mmtoir_command)
    mmtocode_command = 'mmtocode -f caffe -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'caffe_'+pytorch_model_upload_userinfolist[1]+'.py -dw caffe_'+pytorch_model_upload_userinfolist[1]+'.npy'
    print(mmtocode_command)
    val2 = os.system(mmtocode_command)
    mmtomodel_command = 'mmtomodel -f caffe -in caffe_'+pytorch_model_upload_userinfolist[1]+'.py -iw caffe_'+pytorch_model_upload_userinfolist[1]+'.npy -o converted_caffe_model'
    print(mmtomodel_command)
    val3 = os.system(mmtomodel_command)
    print("The model has been converted successfully")
    return html


@app.route('/compiler/v1.0/pytorchmodel_convert_to_caffe_test', methods=['GET', 'POST'])
def pytorchmodel_convert_to_caffe_test():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Caffe Model Converted to MXNET Model Successfully!!!
          </div>
          <div>
          <button onclick="begin_download_params()">Params file download</button>
          <button onclick="begin_download_json()">Json file download</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=convertedmodel-0000.params'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=convertedmodel-symbol.json'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    print(caffe_convert_mxnet_upload_filelist[0])
    print(caffe_convert_mxnet_upload_filelist[1])
    command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    print(command)
    val = os.system(command)
    print(val)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/pytorchmodel_convert_to_mxnet', methods=['GET', 'POST'])
def pytorchmodel_convert_to_mxnet():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Pytorch模型转mxnet模型成功
          </div>
          <div>
          <button onclick="begin_download_params()">Params file download</button>
          <button onclick="begin_download_json()">Json file download</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=pytorch_convert_mxnet-0000.params'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=pytorch_convert_mxnet-symbol.json'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(pytorch_model_upload_userinfolist[0])
    print(pytorch_model_upload_userinfolist[1])
    print(pytorch_model_upload_filelist[0])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n ./save_file/'+pytorch_model_upload_filelist[0]
    print(mmtoir_command)
    val1 = os.system(mmtoir_command)
    mmtocode_command = 'mmtocode -f mxnet -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'mxnet_'+pytorch_model_upload_userinfolist[1]+'.py -dw mxnet_'+pytorch_model_upload_userinfolist[1]+'.npy'
    print(mmtocode_command)
    val2 = os.system(mmtocode_command)
    mmtomodel_command = 'mmtomodel -f mxnet -in mxnet_'+pytorch_model_upload_userinfolist[1]+'.py -iw mxnet_'+pytorch_model_upload_userinfolist[1]+'.npy -o pytorch_convert_mxnet'
    print(mmtomodel_command)
    val3 = os.system(mmtomodel_command)
    print("The model has been converted successfully")
    return html


@app.route('/compiler/v1.0/pytorchmodel_convert_to_mxnet_test', methods=['GET', 'POST'])
def pytorchmodel_convert_to_mxnet_test():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Caffe Model Converted to MXNET Model Successfully!!!
          </div>
          <div>
          <button onclick="begin_download_params()">Params file download</button>
          <button onclick="begin_download_json()">Json file download</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=convertedmodel-0000.params'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=convertedmodel-symbol.json'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    print(caffe_convert_mxnet_upload_filelist[0])
    print(caffe_convert_mxnet_upload_filelist[1])
    command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    print(command)
    val = os.system(command)
    print(val)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/pytorchmodel_convert_to_onnx_test', methods=['GET', 'POST'])
def pytorchmodel_convert_to_onnx_test():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Caffe Model Converted to MXNET Model Successfully!!!
          </div>
          <div>
          <button onclick="begin_download_params()">Params file download</button>
          <button onclick="begin_download_json()">Json file download</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=convertedmodel-0000.params'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=convertedmodel-symbol.json'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    print(caffe_convert_mxnet_upload_filelist[0])
    print(caffe_convert_mxnet_upload_filelist[1])
    command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    print(command)
    val = os.system(command)
    print(val)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/pytorchmodel_convert_to_onnx', methods=['GET', 'POST'])
def pytorchmodel_convert_to_onnx():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Pytorch模型转onnx模型成功
          </div>
          <div>
          <button onclick="begin_download_params()">ONNX模型文件下载</button>
          <!--button onclick="begin_download_json()">Json file download</button-->
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=pytorch_convert_onnx.onnx'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=convertedmodel-symbol.json'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(pytorch_model_upload_userinfolist[0])
    print(pytorch_model_upload_userinfolist[1])
    print(pytorch_model_upload_filelist[0])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n ./save_file/'+pytorch_model_upload_filelist[0]
    print(mmtoir_command)
    val1 = os.system(mmtoir_command)
    mmtocode_command = 'mmtocode -f onnx -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'onnx_'+pytorch_model_upload_userinfolist[1]+'.py -dw onnx_'+pytorch_model_upload_userinfolist[1]+'.npy'
    print(mmtocode_command)
    val2 = os.system(mmtocode_command)
    mmtomodel_command = 'mmtomodel -f onnx -in onnx_'+pytorch_model_upload_userinfolist[1]+'.py -iw onnx_'+pytorch_model_upload_userinfolist[1]+'.npy -o pytorch_convert_onnx.onnx'
    print(mmtomodel_command)
    val3 = os.system(mmtomodel_command)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/tensorflowmodel_convert_to_pytorch', methods=['GET', 'POST'])
def tensorflowmodel_convert_to_pytorch():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Tensorflow模型转IR成功
          </div>
          <div>
          <button onclick="begin_download_params()">JSON文件下载</button>
          <button onclick="begin_download_json()">pb文件下载</button>
          <button onclick="begin_download_npy()">npy文件下载</button>
          <button onclick="begin_optimization()">编译优化</button>
          <button onclick="no_optimization()">未编译优化</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=converted.json'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=converted.pb'
            }
            function begin_download_npy() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=converted.npy'
            }
            function begin_optimization() {
                window.location.href='http://$HOST:5005/compiler/v1.0/ir_fpga_optimization'
            }
            function no_optimization() {
                window.location.href='http://$HOST:5005/compiler/v1.0/ir_fpga_no_optimization'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(tensorflow_model_upload_userinfolist[0])
    print(tensorflow_model_upload_userinfolist[1])
    print(tensorflow_model_upload_filelist)
    #print(tensorflow_model_upload_filelist[1])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    #mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n '+pytorch_model_upload_filelist[0]
    #print(mmtoir_command)
    #val1 = os.system(mmtoir_command)
    #mmtocode_command = 'mmtocode -f tensorflow -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'tensorflow_'+pytorch_model_upload_filelist[0]+'.py'
    #print(mmtocode_command)
    #val2 = os.system(mmtocode_command)
    #mmtomodel_command = 'mmtomodel -f tensorflow -in tensorflow_'+pytorch_model_upload_filelist[0]+'.py -iw '+pytorch_model_upload_userinfolist[1]+'.npy -o converted_model.ckpt'
    #print(mmtomodel_command)
    #val3 = os.system(mmtomodel_command)
    #convert_command = 'mmconvert -sf tensorflow -in ./save_file/'+tensorflow_model_upload_filelist[0]+' -iw ./save_file/'+tensorflow_model_upload_filelist[1]+' --dstNodeName '+tensorflow_model_upload_userinfolist[1]+' -df pytorch -om tf_convert_pytorch.pth'
    #print(convert_command)
    #val = os.system(convert_command)
    #print(val)
    mmtoir_command = 'python3 -m mmdnn.conversion._script.convertToIR -f tensorflow -n ./models/tf/saved_model.pb -w ./models/tf/saved_model.pb --inputShape 1,800,800 -o converted'
    print(mmtoir_command)
    val = os.system(mmtoir_command)
    print(val)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/tensorflowmodel_convert_to_caffe', methods=['GET', 'POST'])
def tensorflowmodel_convert_to_caffe():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Tensorflow模型转caffe模型成功
          </div>
          <div>
          <button onclick="begin_download_params()">caffemodel文件下载</button>
          <button onclick="begin_download_json()">prototxt文件下载</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=tf_convert_caffe.caffemodel'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=tf_convert_caffe.prototxt'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(tensorflow_model_upload_userinfolist[0])
    print(tensorflow_model_upload_userinfolist[1])
    print(tensorflow_model_upload_filelist[0])
    print(tensorflow_model_upload_filelist[1])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    #mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n '+pytorch_model_upload_filelist[0]
    #print(mmtoir_command)
    #val1 = os.system(mmtoir_command)
    #mmtocode_command = 'mmtocode -f tensorflow -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'tensorflow_'+pytorch_model_upload_filelist[0]+'.py'
    #print(mmtocode_command)
    #val2 = os.system(mmtocode_command)
    #mmtomodel_command = 'mmtomodel -f tensorflow -in tensorflow_'+pytorch_model_upload_filelist[0]+'.py -iw '+pytorch_model_upload_userinfolist[1]+'.npy -o converted_model.ckpt'
    #print(mmtomodel_command)
    #val3 = os.system(mmtomodel_command)
    convert_command = 'mmconvert -sf tensorflow -in ./save_file/'+tensorflow_model_upload_filelist[0]+' -iw ./save_file/'+tensorflow_model_upload_filelist[1]+' --dstNodeName '+tensorflow_model_upload_userinfolist[1]+' -df caffe -om tf_convert_caffe'
    print(convert_command)
    val = os.system(convert_command)
    print(val)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/tensorflowmodel_convert_to_mxnet', methods=['GET', 'POST'])
def tensorflowmodel_convert_to_mxnet():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Tensorflow模型转mxnet模型成功
          </div>
          <div>
          <button onclick="begin_download_params()">Params文件下载</button>
          <button onclick="begin_download_json()">Json文件下载</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=tf_convert_mxnet-0000.params'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=tf_convert_mxnet-symbol.json'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(tensorflow_model_upload_userinfolist[0])
    print(tensorflow_model_upload_userinfolist[1])
    print(tensorflow_model_upload_filelist[0])
    print(tensorflow_model_upload_filelist[1])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    #mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n '+pytorch_model_upload_filelist[0]
    #print(mmtoir_command)
    #val1 = os.system(mmtoir_command)
    #mmtocode_command = 'mmtocode -f tensorflow -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'tensorflow_'+pytorch_model_upload_filelist[0]+'.py'
    #print(mmtocode_command)
    #val2 = os.system(mmtocode_command)
    #mmtomodel_command = 'mmtomodel -f tensorflow -in tensorflow_'+pytorch_model_upload_filelist[0]+'.py -iw '+pytorch_model_upload_userinfolist[1]+'.npy -o converted_model.ckpt'
    #print(mmtomodel_command)
    #val3 = os.system(mmtomodel_command)
    convert_command = 'mmconvert -sf tensorflow -in ./save_file/'+tensorflow_model_upload_filelist[0]+' -iw ./save_file/'+tensorflow_model_upload_filelist[1]+' --dstNodeName '+tensorflow_model_upload_userinfolist[1]+' -df mxnet -om tf_convert_mxnet'
    print(convert_command)
    val = os.system(convert_command)
    print(val)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/tensorflowmodel_convert_to_onnx', methods=['GET', 'POST'])
def tensorflowmodel_convert_to_onnx():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Tensorflow模型转onnx模型成功
          </div>
          <div>
          <button onclick="begin_download_params()">ONNX模型文件下载</button>
          <!--button onclick="begin_download_json()">Json file download</button-->
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=tf_convert_onnx.onnx'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=convertedmodel-symbol.json'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(tensorflow_model_upload_userinfolist[0])
    print(tensorflow_model_upload_userinfolist[1])
    print(tensorflow_model_upload_filelist[0])
    print(tensorflow_model_upload_filelist[1])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    #mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n '+pytorch_model_upload_filelist[0]
    #print(mmtoir_command)
    #val1 = os.system(mmtoir_command)
    #mmtocode_command = 'mmtocode -f tensorflow -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'tensorflow_'+pytorch_model_upload_filelist[0]+'.py'
    #print(mmtocode_command)
    #val2 = os.system(mmtocode_command)
    #mmtomodel_command = 'mmtomodel -f tensorflow -in tensorflow_'+pytorch_model_upload_filelist[0]+'.py -iw '+pytorch_model_upload_userinfolist[1]+'.npy -o converted_model.ckpt'
    #print(mmtomodel_command)
    #val3 = os.system(mmtomodel_command)
    convert_command = 'mmconvert -sf tensorflow -in ./save_file/'+tensorflow_model_upload_filelist[0]+' -iw ./save_file/'+tensorflow_model_upload_filelist[1]+' --dstNodeName '+tensorflow_model_upload_userinfolist[1]+' -df onnx -om tf_convert_onnx.onnx'
    print(convert_command)
    val = os.system(convert_command)
    print(val)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/caffemodel_convert_to_pytorch', methods=['GET', 'POST'])
def caffemodel_convert_to_pytorch():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Caffe模型转IR成功
          </div>
          <div>
          <button onclick="begin_download_params()">JSON文件下载</button>
          <button onclick="begin_download_json()">pb文件下载</button>
          <button onclick="begin_download_npy()">npy文件下载</button>
          <button onclick="begin_optimization()">编译优化</button>
          <button onclick="no_optimization()">未编译优化</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=converted.json'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=converted.pb'
            }
            function begin_download_npy() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=converted.npy'
            }
            function begin_optimization() {
                window.location.href='http://$HOST:5005/compiler/v1.0/ir_fpga_optimization'
            }
            function no_optimization() {
                window.location.href='http://$HOST:5005/compiler/v1.0/ir_fpga_no_optimization'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(caffe_model_upload_userinfolist[0])
    print(caffe_model_upload_userinfolist[1])
    print(caffe_model_upload_filelist[0])
    print(caffe_model_upload_filelist[1])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    #mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n '+pytorch_model_upload_filelist[0]
    #print(mmtoir_command)
    #val1 = os.system(mmtoir_command)
    #mmtocode_command = 'mmtocode -f tensorflow -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'tensorflow_'+pytorch_model_upload_filelist[0]+'.py'
    #print(mmtocode_command)
    #val2 = os.system(mmtocode_command)
    #mmtomodel_command = 'mmtomodel -f tensorflow -in tensorflow_'+pytorch_model_upload_filelist[0]+'.py -iw '+pytorch_model_upload_userinfolist[1]+'.npy -o converted_model.ckpt'
    #print(mmtomodel_command)
    #val3 = os.system(mmtomodel_command)
    #convert_command = 'mmconvert -sf caffe -in ./save_file/'+caffe_model_upload_filelist[0]+' -iw ./save_file/'+caffe_model_upload_filelist[1]+' -df pytorch -om caffe_convert_pytorch.pth'
    #print(convert_command)
    #val = os.system(convert_command)
    #print(val)
    mmtoir_command = 'python3 -m mmdnn.conversion._script.convertToIR -f caffe -n ./models/caffe/cfpt.prototxt -w ./models/caffe/cfmodel.caffemodel --inputShape 1,800,800 -o converted'
    print(mmtoir_command)
    val = os.system(mmtoir_command)
    print(val)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/caffemodel_convert_to_tensorflow', methods=['GET', 'POST'])
def caffemodel_convert_to_tensorflow():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Caffe模型转Tensorflow模型成功
          </div>
          <div>
          <button onclick="begin_download_params()">tensorflow模型文件下载</button>
          <!--button onclick="begin_download_json()">Json file download</button-->
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=tensorflow.tar.gz'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=convertedmodel-symbol.json'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(caffe_model_upload_userinfolist[0])
    print(caffe_model_upload_userinfolist[1])
    print(caffe_model_upload_filelist[0])
    print(caffe_model_upload_filelist[1])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    #mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n '+pytorch_model_upload_filelist[0]
    #print(mmtoir_command)
    #val1 = os.system(mmtoir_command)
    #mmtocode_command = 'mmtocode -f tensorflow -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'tensorflow_'+pytorch_model_upload_filelist[0]+'.py'
    #print(mmtocode_command)
    #val2 = os.system(mmtocode_command)
    #mmtomodel_command = 'mmtomodel -f tensorflow -in tensorflow_'+pytorch_model_upload_filelist[0]+'.py -iw '+pytorch_model_upload_userinfolist[1]+'.npy -o converted_model.ckpt'
    #print(mmtomodel_command)
    #val3 = os.system(mmtomodel_command)
    convert_command = 'rm -f tensorflow.tar.gz && rm -r caffe_convert_tensorflow.ckpt && mmconvert -sf caffe -in ./save_file/'+caffe_model_upload_filelist[0]+' -iw ./save_file/'+caffe_model_upload_filelist[1]+' -df tensorflow -om caffe_convert_tensorflow.ckpt && tar -zcvf tensorflow.tar.gz ./caffe_convert_tensorflow.ckpt'
    print(convert_command)
    val = os.system(convert_command)
    print(val)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/caffemodel_convert_to_mxnet', methods=['GET', 'POST'])
def caffemodel_convert_to_mxnet():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Caffe模型转mxnet模型成功
          </div>
          <div>
          <button onclick="begin_download_params()">Params file download</button>
          <button onclick="begin_download_json()">Json file download</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=caffe_convert_mxnet-0000.params'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=caffe_convert_mxnet-symbol.json'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(caffe_model_upload_userinfolist[0])
    print(caffe_model_upload_userinfolist[1])
    print(caffe_model_upload_filelist[0])
    print(caffe_model_upload_filelist[1])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    #mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n '+pytorch_model_upload_filelist[0]
    #print(mmtoir_command)
    #val1 = os.system(mmtoir_command)
    #mmtocode_command = 'mmtocode -f tensorflow -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'tensorflow_'+pytorch_model_upload_filelist[0]+'.py'
    #print(mmtocode_command)
    #val2 = os.system(mmtocode_command)
    #mmtomodel_command = 'mmtomodel -f tensorflow -in tensorflow_'+pytorch_model_upload_filelist[0]+'.py -iw '+pytorch_model_upload_userinfolist[1]+'.npy -o converted_model.ckpt'
    #print(mmtomodel_command)
    #val3 = os.system(mmtomodel_command)
    convert_command = 'mmconvert -sf caffe -in ./save_file/'+caffe_model_upload_filelist[0]+' -iw ./save_file/'+caffe_model_upload_filelist[1]+' -df mxnet -om caffe_convert_mxnet'
    print(convert_command)
    val = os.system(convert_command)
    print(val)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/caffemodel_convert_to_onnx', methods=['GET', 'POST'])
def caffemodel_convert_to_onnx():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Caffe模型转onnx模型成功
          </div>
          <div>
          <button onclick="begin_download_params()">ONNX模型文件下载</button>
          <!--button onclick="begin_download_json()">Json file download</button-->
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=caffe_convert_onnx.onnx'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=convertedmodel-symbol.json'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(caffe_model_upload_userinfolist[0])
    print(caffe_model_upload_userinfolist[1])
    print(caffe_model_upload_filelist[0])
    print(caffe_model_upload_filelist[1])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    #mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n '+pytorch_model_upload_filelist[0]
    #print(mmtoir_command)
    #val1 = os.system(mmtoir_command)
    #mmtocode_command = 'mmtocode -f tensorflow -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'tensorflow_'+pytorch_model_upload_filelist[0]+'.py'
    #print(mmtocode_command)
    #val2 = os.system(mmtocode_command)
    #mmtomodel_command = 'mmtomodel -f tensorflow -in tensorflow_'+pytorch_model_upload_filelist[0]+'.py -iw '+pytorch_model_upload_userinfolist[1]+'.npy -o converted_model.ckpt'
    #print(mmtomodel_command)
    #val3 = os.system(mmtomodel_command)
    convert_command = 'mmconvert -sf caffe -in ./save_file/'+caffe_model_upload_filelist[0]+' -iw ./save_file/'+caffe_model_upload_filelist[1]+' -df onnx -om caffe_convert_onnx.onnx'
    print(convert_command)
    val = os.system(convert_command)
    print(val)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/mxnetmodel_convert_to_pytorch', methods=['GET', 'POST'])
def mxnetmodel_convert_to_pytorch():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Mxnet模型转IR成功
          </div>
          <div>
          <button onclick="begin_download_params()">JSON文件下载</button>
          <button onclick="begin_download_json()">pb文件下载</button>
          <button onclick="begin_download_npy()">npy文件下载</button>
          <button onclick="begin_optimization()">编译优化</button>
          <button onclick="no_optimization()">未编译优化</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=converted.json'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=converted.pb'
            }
            function begin_download_npy() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=converted.npy'
            }
            function begin_optimization() {
                window.location.href='http://$HOST:5005/compiler/v1.0/ir_fpga_optimization'
            }
            function no_optimization() {
                window.location.href='http://$HOST:5005/compiler/v1.0/ir_fpga_no_optimization'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(mxnet_model_upload_userinfolist[0])
    print(mxnet_model_upload_userinfolist[1])
    print(mxnet_model_upload_filelist[0])
    print(mxnet_model_upload_filelist[1])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    #mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n '+pytorch_model_upload_filelist[0]
    #print(mmtoir_command)
    #val1 = os.system(mmtoir_command)
    #mmtocode_command = 'mmtocode -f tensorflow -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'tensorflow_'+pytorch_model_upload_filelist[0]+'.py'
    #print(mmtocode_command)
    #val2 = os.system(mmtocode_command)
    #mmtomodel_command = 'mmtomodel -f tensorflow -in tensorflow_'+pytorch_model_upload_filelist[0]+'.py -iw '+pytorch_model_upload_userinfolist[1]+'.npy -o converted_model.ckpt'
    #print(mmtomodel_command)
    #val3 = os.system(mmtomodel_command)
    #convert_command = 'mmconvert -sf mxnet -in ./save_file/'+mxnet_model_upload_filelist[0]+' -iw ./save_file/'+mxnet_model_upload_filelist[1]+' -df pytorch -om mxnet_convert_pytorch.pth --inputShape '+mxnet_model_upload_userinfolist[0]
    #print(convert_command)
    #val = os.system(convert_command)
    #print(val)
    mmtoir_command = 'python3 -m mmdnn.conversion._script.convertToIR -f mxnet -n ./models/mxnet/mxmodel.py -w ./models/mxnet/mxweights.params --inputShape 1,800,800 -o converted'
    print(mmtoir_command)
    val = os.system(mmtoir_command)
    print(val)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/mxnetmodel_convert_to_tensorflow', methods=['GET', 'POST'])
def mxnetmodel_convert_to_tensorflow():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Mxnet模型转tensorflow模型成功
          </div>
          <div>
          <button onclick="begin_download_params()">Tensorflow模型文件下载</button>
          <!--button onclick="begin_download_json()">Json file download</button-->
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=mxnet_to_tensorflow.tar.gz'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=convertedmodel-symbol.json'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(mxnet_model_upload_userinfolist[0])
    print(mxnet_model_upload_userinfolist[1])
    print(mxnet_model_upload_filelist[0])
    print(mxnet_model_upload_filelist[1])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    #mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n '+pytorch_model_upload_filelist[0]
    #print(mmtoir_command)
    #val1 = os.system(mmtoir_command)
    #mmtocode_command = 'mmtocode -f tensorflow -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'tensorflow_'+pytorch_model_upload_filelist[0]+'.py'
    #print(mmtocode_command)
    #val2 = os.system(mmtocode_command)
    #mmtomodel_command = 'mmtomodel -f tensorflow -in tensorflow_'+pytorch_model_upload_filelist[0]+'.py -iw '+pytorch_model_upload_userinfolist[1]+'.npy -o converted_model.ckpt'
    #print(mmtomodel_command)
    #val3 = os.system(mmtomodel_command)
    convert_command = 'rm -f mxnet_to_tensorflow.tar.gz && rm -r mxnet_convert_tensorflow.ckpt && mmconvert -sf mxnet -in ./save_file/'+mxnet_model_upload_filelist[0]+' -iw ./save_file/'+mxnet_model_upload_filelist[1]+' -df tensorflow -om mxnet_convert_tensorflow.ckpt --inputShape '+mxnet_model_upload_userinfolist[0]+' && tar -zcvf mxnet_to_tensorflow.tar.gz ./mxnet_convert_tensorflow.ckpt'
    print(convert_command)
    val = os.system(convert_command)
    print(val)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/mxnetmodel_convert_to_caffe', methods=['GET', 'POST'])
def mxnetmodel_convert_to_caffe():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Mxnet模型转caffe模型成功
          </div>
          <div>
          <button onclick="begin_download_params()">caffemodel文件下载</button>
          <button onclick="begin_download_json()">prototxt文件下载</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=mxnet_convert_caffe.caffemodel'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=mxnet_convert_caffe.prototxt'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(mxnet_model_upload_userinfolist[0])
    print(mxnet_model_upload_userinfolist[1])
    print(mxnet_model_upload_filelist[0])
    print(mxnet_model_upload_filelist[1])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    #mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n '+pytorch_model_upload_filelist[0]
    #print(mmtoir_command)
    #val1 = os.system(mmtoir_command)
    #mmtocode_command = 'mmtocode -f tensorflow -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'tensorflow_'+pytorch_model_upload_filelist[0]+'.py'
    #print(mmtocode_command)
    #val2 = os.system(mmtocode_command)
    #mmtomodel_command = 'mmtomodel -f tensorflow -in tensorflow_'+pytorch_model_upload_filelist[0]+'.py -iw '+pytorch_model_upload_userinfolist[1]+'.npy -o converted_model.ckpt'
    #print(mmtomodel_command)
    #val3 = os.system(mmtomodel_command)
    convert_command = 'mmconvert -sf mxnet -in ./save_file/'+mxnet_model_upload_filelist[0]+' -iw ./save_file/'+mxnet_model_upload_filelist[1]+' -df caffe -om mxnet_convert_caffe --inputShape '+mxnet_model_upload_userinfolist[0]
    print(convert_command)
    val = os.system(convert_command)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/mxnetmodel_convert_to_onnx', methods=['GET', 'POST'])
def mxnetmodel_convert_to_onnx():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Mxnet模型转onnx模型成功
          </div>
          <div>
          <button onclick="begin_download_params()">ONNX模型文件下载</button>
          <!--button onclick="begin_download_json()">Json file download</button-->
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=mxnet_convert_onnx.onnx'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=convertedmodel-symbol.json'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    print(mxnet_model_upload_userinfolist[0])
    print(mxnet_model_upload_userinfolist[1])
    print(mxnet_model_upload_filelist[0])
    print(mxnet_model_upload_filelist[1])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
    #val = os.system(command)
    #print(val)
    #mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n '+pytorch_model_upload_filelist[0]
    #print(mmtoir_command)
    #val1 = os.system(mmtoir_command)
    #mmtocode_command = 'mmtocode -f tensorflow -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'tensorflow_'+pytorch_model_upload_filelist[0]+'.py'
    #print(mmtocode_command)
    #val2 = os.system(mmtocode_command)
    #mmtomodel_command = 'mmtomodel -f tensorflow -in tensorflow_'+pytorch_model_upload_filelist[0]+'.py -iw '+pytorch_model_upload_userinfolist[1]+'.npy -o converted_model.ckpt'
    #print(mmtomodel_command)
    #val3 = os.system(mmtomodel_command)
    convert_command = 'mmconvert -sf mxnet -in ./save_file/'+mxnet_model_upload_filelist[0]+' -iw ./save_file/'+mxnet_model_upload_filelist[1]+' -df onnx -om mxnet_convert_onnx.onnx --inputShape '+mxnet_model_upload_userinfolist[0]
    print(convert_command)
    val = os.system(convert_command)
    print("The model has been converted successfully")
    return html

@app.route('/compiler/v1.0/onnxmodel_convert_to_ir', methods=['GET', 'POST'])
def onnxmodel_convert_to_ir():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          ONNX模型转IR成功
          </div>
          <div>
          <button onclick="begin_download_params()">JSON文件下载</button>
          <button onclick="begin_download_json()">pb文件下载</button>
          <button onclick="begin_download_npy()">npy文件下载</button>
          <button onclick="begin_optimization()">编译优化</button>
          <button onclick="no_optimization()">未编译优化</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://$HOST:$PORT/mxnet_params_file_download?fileId=converted.json'
            }
            function begin_download_json() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=converted.pb'
            }
            function begin_download_npy() {
                window.location.href='http://$HOST:$PORT/mxnet_json_file_download?fileId=converted.npy'
            }
            function begin_optimization() {
                window.location.href='http://$HOST:5005/compiler/v1.0/ir_fpga_optimization'
            }
            function no_optimization() {
                window.location.href='http://$HOST:5005/compiler/v1.0/ir_fpga_no_optimization'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    #print(caffe_convert_mxnet_upload_filelist[0])
    #print(caffe_convert_mxnet_upload_filelist[1])
    #print(mxnet_model_upload_userinfolist[0])
    #print(mxnet_model_upload_userinfolist[1])
    #print(mxnet_model_upload_filelist[0])
    #print(mxnet_model_upload_filelist[1])
    #command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    #print(command)
   # #val = os.system(command)
    #print(val)
    #mmtoir_command = 'mmtoir -f pytorch -d '+pytorch_model_upload_userinfolist[1]+' --inputShape '+pytorch_model_upload_userinfolist[0]+' -n '+pytorch_model_upload_filelist[0]
    #print(mmtoir_command)
    #val1 = os.system(mmtoir_command)
    #mmtocode_command = 'mmtocode -f tensorflow -n '+pytorch_model_upload_userinfolist[1]+'.pb --IRWeightPath '+pytorch_model_upload_userinfolist[1]+'.npy --dstModelPath '+'tensorflow_'+pytorch_model_upload_filelist[0]+'.py'
    #print(mmtocode_command)
    #val2 = os.system(mmtocode_command)
    #mmtomodel_command = 'mmtomodel -f tensorflow -in tensorflow_'+pytorch_model_upload_filelist[0]+'.py -iw '+pytorch_model_upload_userinfolist[1]+'.npy -o converted_model.ckpt'
    #print(mmtomodel_command)
    #val3 = os.system(mmtomodel_command)
    #convert_command = 'mmconvert -sf mxnet -in ./save_file/'+mxnet_model_upload_filelist[0]+' -iw ./save_file/'+mxnet_model_upload_filelist[1]+' -df onnx -om mxnet_convert_onnx.onnx --inputShape '+mxnet_model_upload_userinfolist[0]
    #print(convert_command)
    #val = os.system(convert_command)
    mmtoir_command = 'python3 -m mmdnn.conversion._script.convertToIR -f onnx -n ./models/onnx/YOLO.onnx -w ./models/onnx/YOLO.onnx --inputShape 1,800,800 -o converted'
    print(mmtoir_command)
    val = os.system(mmtoir_command)
    print(val)
    print("The model has been converted successfully")
    return html


########################################################################################################################################################################

@app.route('/compiler/v1.0/convert_mxnet_to_onnx')
def convert_mxnet_to_onnx():
    """
    ▒~V~R~T▒~V~R~[~^▒~V~R~@个▒~V~R~Q页端▒~V~R~O~P交▒~V~R~Z~D页▒~V~R~]▒~V~R
    :return:
    """
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>

          <form action = "http://$HOST:$PORT/compiler/v1.0/flops/file_upload" method = "POST"
             enctype = "multipart/form-data">
             <input type = "file" name = "file" />
             <input type = "submit"/>
          </form>

       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    return html



@app.route('/compiler/v1.0/flops_compute')
def flops_compute():
    """
    ▒~T▒~[~^▒~@个▒~Q页端▒~O~P交▒~Z~D页▒~]▒
    :return:
    """
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>

          <form action = "http://$HOST:$PORT/compiler/v1.0/flops/file_upload" method = "POST"
             enctype = "multipart/form-data">
             <input type = "file" name = "file" />
             <input type = "submit"/>
          </form>

       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    return html

def allowed_file(filename):
    """
    ▒~@▒~L▒~V~G件▒~P~M尾▒~@▒~X▒▒~P▒满足▒| ▒▒~O▒~A▒~B
    :param filename:
    :return:
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/compiler/v1.0/flops/file_upload', methods=['GET', 'POST'])
def upload_file():
    """
    ▒~J▒| ▒~V~G件▒~H▒save_file▒~V~G件夹
    以requests▒~J▒| 举▒~K
    wiht open('路▒~D','rb') as file_obj:
        rsp = requests.post('http://localhost:5000/upload,files={'file':file_obj})
        print(rsp.text) --> file uploaded successfully
    """
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #return 'file uploaded successfully'
        model_convert()
        return 'file '+filename+' uploaded successfully'
    return "file uploaded Fail"


@app.route('/compiler/v1.0/model_visualization')
def model_visualization():
    """
    ▒~V~R~T▒~V~R~[~^▒~V~R~@个▒~V~R~Q页端▒~V~R~O~P交▒~V~R~Z~D页▒~V~R~]▒~V~R
    :return:
    """
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>

          <form id="my_frame" action = "http://$HOST:$PORT/compiler/v1.0/flops/model_upload" method = "POST"
             enctype = "multipart/form-data">
             <input type = "file" name = "file" />
             <input type = "submit"/>
          </form>
          <iframe class="hidden_frame" name='hidden_frame' id="hidden_frame" style='display: none'></iframe>
          <script>
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    return html


@app.route('/compiler/v1.0/model_convert')
def model_Convert():
    caffe_convert_mxnet_upload_filelist.clear()
    """
    ▒~V~R~V~R~T▒~V~R~V~R~[~^▒~V~R~V~R~@个▒~V~R~V~R~Q页端▒~V~R~V~R~O~P交▒~V~R~V~R~Z~D页▒~V~R~V~R~]▒~V~R~V~R
    :return:
    """
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>

          <form id="my_frame1" action = "http://$HOST:$PORT/compiler/v1.0/flops/caffemodel_upload" method = "POST"
             enctype = "multipart/form-data">
             <input type = "file" name = "file" />
             <input type = "file" name = "file" />
             <input type = "submit"/>
          </form>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    return html


@app.route('/compiler/v1.0/flops/caffemodel_upload_test', methods=['GET', 'POST'])
def caffeupload_file_test():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Model files Uploaded Successfully!!!
          </div>
          <div>
          <button onclick="begin_convert_to_mxnet()">Begin Convert to Mxnet</button>
          </div>
          <div>
          <button onclick="begin_convert_to_pytorch()">Begin Convert to Pytorch</button>
          </div>
          <div>
          <button onclick="begin_convert_to_tensorflow()">Begin Convert to Tensorflow</button>
          </div>
          <div>
          <button onclick="begin_convert_to_onnx()">Begin Convert to onnx</button>
          </div>
          <script>
            function begin_convert_to_mxnet() {
                window.location.href='http://172.20.0.58:5000/compiler/v1.0/caffemodel_convert_to_mxnet';
            }
            function begin_convert_to_pytorch() {
                window.location.href='http://172.20.0.58:5000/compiler/v1.0/caffemodel_convert_to_pytorch';
            }
            function begin_convert_to_tensorflow() {
                window.location.href='http://172.20.0.58:5000/compiler/v1.0/caffemodel_convert_to_tensorflow';
            }
            function begin_convert_to_onnx() {
                window.location.href='http://172.20.0.58:5000/compiler/v1.0/caffemodel_convert_to_onnx';
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    """
    ▒~V~R~J▒~V~R| ▒~V~R~V~G件▒~V~R~H▒~V~Rsave_file▒~V~R~V~G件夹
    以requests▒~V~R~J▒~V~R| 举▒~V~R~K
    wiht open('路▒~V~R~D','rb') as file_obj:
        rsp = requests.post('http://localhost:5000/upload,files={'file':file_obj})
        print(rsp.text) --> file uploaded successfully
    """
    for file in request.files.getlist('file'):
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            caffe_convert_mxnet_upload_filelist.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename+" upload successfully")
    #return 'files uploaded successfully'
    return html
    #if 'file' not in request.files:
    #    return "No file part"
    #file = request.files['file']
    #if file.filename == '':
    #    return 'No selected file'
    #if file and allowed_file(file.filename):
    #    filename = secure_filename(file.filename)
    #    print(filename)
    #    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #    return 'file uploaded successfully'
    #    model_convert()
        #return 'file '+filename+' uploaded successfully'
    #return "file uploaded Fail"

@app.route('/compiler/v1.0/caffemodel_convert_to_mxnet_test', methods=['GET', 'POST'])
def caffemodel_convert_to_mxnet_test():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Caffe Model Converted to MXNET Model Successfully!!!
          </div>
          <div>
          <button onclick="begin_download_params()">Params file download</button>
          <button onclick="begin_download_json()">Json file download</button>
          </div>
          <script>
            function begin_download_params() {
                window.location.href='http://172.20.0.58:5000/mxnet_params_file_download?fileId=convertedmodel-0000.params'
            }
            function begin_download_json() {
                window.location.href='http://172.20.0.58:5000/mxnet_json_file_download?fileId=convertedmodel-symbol.json'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    print(caffe_convert_mxnet_upload_filelist[0])
    print(caffe_convert_mxnet_upload_filelist[1])
    command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df mxnet -om ./mxnet/convertedmodel'
    print(command)
    val = os.system(command)
    print(val)
    print("The model has been converted successfully")
    return html
    

@app.route('/compiler/v1.0/caffemodel_convert_to_pytorch_test', methods=['GET', 'POST'])
def caffemodel_convert_to_pytorch_test():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Caffe Model Converted to Pytorch Model Successfully!!!
          </div>
          <div>
          <button onclick="begin_download_pth()">Pth file download</button>
          </div>
          <script>
            function begin_download_pth() {
                window.location.href='http://172.20.0.58:5000/pytorch_pth_file_download?fileId=convertedmodel.pth'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    print(caffe_convert_mxnet_upload_filelist[0])
    print(caffe_convert_mxnet_upload_filelist[1])
    command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df pytorch -om ./pytorch/convertedmodel.pth'
    print(command)
    val = os.system(command)
    print(val)
    print("The model has been converted successfully")
    return html


@app.route('/compiler/v1.0/caffemodel_convert_to_tensorflow_test', methods=['GET', 'POST'])
def caffemodel_convert_to_tensorflow_test():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Caffe Model Converted to Tensorflow Model Successfully!!!
          </div>
          <div>
          <button onclick="begin_download_TF_model()">Tensorflow_model_file download</button>
          </div>
          <script>
            function begin_download_TF_model() {
                window.location.href='http://192.168.200.178:5000/tensorflow_model_file_download?fileId=convertedmodel.tar'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    print(caffe_convert_mxnet_upload_filelist[0])
    print(caffe_convert_mxnet_upload_filelist[1])
    rm_command = 'rm -rf ./tensorflow/*'
    print(rm_command)
    var_rm = os.system(rm_command)
    print(var_rm)
    command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df tensorflow -om ./tensorflow/convertedmodel'
    print(command)
    val = os.system(command)
    print(val)
    tar_command = 'cd ./tensorflow && tar -zcvf convertedmodel.tar convertedmodel'
    print(tar_command)
    var_tar = os.system(tar_command)
    print(var_tar)
    print("The model has been converted successfully")
    return html


@app.route('/compiler/v1.0/caffemodel_convert_to_onnx_test', methods=['GET', 'POST'])
def caffemodel_convert_to_onnx_test():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          Caffe Model Converted to onnx Model Successfully!!!
          </div>
          <div>
          <button onclick="begin_download_onnx()">onnx file download</button>
          </div>
          <script>
            function begin_download_onnx() {
                window.location.href='http://172.20.0.58:5001/onnx_model_file_download?fileId=convertedmodel.onnx'
            }
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})

    print(caffe_convert_mxnet_upload_filelist[0])
    print(caffe_convert_mxnet_upload_filelist[1])
    command = 'mmconvert -sf caffe -in ./save_file/'+caffe_convert_mxnet_upload_filelist[0] + ' -iw ./save_file/'+caffe_convert_mxnet_upload_filelist[1] +' -df onnx -om ./onnx/convertedmodel.onnx'
    print(command)
    val = os.system(command)
    print(val)
    print("The model has been converted successfully")
    return html






@app.route('/compiler/v1.0/flops/caffemodel2_upload', methods=['GET', 'POST'])
def caffeupload_file2():
    """
    ▒~V~R~V~R~J▒~V~R~V~R| ▒~V~R~V~R~V~G件▒~V~R~V~R~H▒~V~R~V~Rsave_file▒~V~R~V~R~V~G件夹
    以requests▒~V~R~V~R~J▒~V~R~V~R| 举▒~V~R~V~R~K
    wiht open('路▒~V~R~V~R~D','rb') as file_obj:
        rsp = requests.post('http://localhost:5000/upload,files={'file':file_obj})
        print(rsp.text) --> file uploaded successfully
    """
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #    return 'file uploaded successfully'
    #    model_convert()
        #return 'file '+filename+' uploaded successfully'
        return
    return "file uploaded Fail"






@app.route('/compiler/v1.0/flops/model_upload', methods=['GET', 'POST'])
def upload_model():
    html = Template("""
    <!DOCTYPE html>
    <html>
       <body>
          <div>
          <button onclick="Visualization()">Model Uploaded Successfully!!!</button>
          </div>
          <script>
            function Visualization() {
                window.location.href='http://192.168.200.178:8081';
            }
            window.onload=Visualization;
          </script>
       </body>
    </html>
    """)
    html = html.substitute({"HOST": HOST, "PORT": PORT})
    """
    ▒~V~R~J▒~V~R| ▒~V~R~V~G件▒~V~R~H▒~V~Rsave_file▒~V~R~V~G件夹
    以requests▒~V~R~J▒~V~R| 举▒~V~R~K
    wiht open('路▒~V~R~D','rb') as file_obj:
        rsp = requests.post('http://localhost:5000/upload,files={'file':file_obj})
        print(rsp.text) --> file uploaded successfully
    """
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #return 'file uploaded successfully'
        model_Visualization(filename)
        return html
        #return render_template('http://10.168.103.104:8081')
        #return 'model '+filename+' uploaded successfully'
    return "model uploaded Fail"


def allowed_file(filename):
    """
    ▒~V~R~@▒~V~R~L▒~V~R~V~G件▒~V~R~P~M尾▒~V~R~@▒~V~R~X▒~V~R▒~V~R~P▒~V~R满足▒~V~R| ▒~V~R▒~V~R~O▒~V~R~A▒~V~R~B
    :param filename:
    :return:
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def model_convert():
    #result = db.query.filter(Article.age == '20').first()
    #result = db.session.query(User).first()
    #result.status = 'The model has been converted successfully'
    #db.session.commit()
    print("The model has been convered successfully")


def model_Visualization(filename):
    #result = db.query.filter(Article.age == '20').first()
    command = 'python3 test_netron.py ./save_file/'+filename+' &'
    print(command)
    #result = db.session.query(User).first()
    #result.status = 'The model has been visualized successfully'
    #db.session.commit()
    value = os.system('./test_kill.sh')
    print(value)
    #val = os.system('python test_netron.py super_resolution_0.2.onnx &')
    val = os.system(command)
    print(val)
    print("The model has been visualized successfully")




@app.route("/download")
def download_file():
    """
    ▒~K载src_file▒~[▒▒~U▒~K▒~]▒▒~Z~D▒~V~G件
    eg▒~Z▒~K载▒~S▒~I~M▒~[▒▒~U▒~K▒~]▒▒~Z~D123.tar ▒~V~G件▒~Leg:http://localhost:5000/download?fileId=123.tar
    :return:
    """
    file_name = request.args.get('fileId')
    file_path = os.path.join(pwd,'src_file',file_name)
    if os.path.isfile(file_path):
        return send_file(file_path,as_attachment=True)
    else:
        return "The downloaded file does not exist"


@app.route("/mxnet_params_file_download")
def mxnet_params_file_download():
    """
    ▒~V~R~K载src_file▒~V~R~[▒~V~R▒~V~R~U▒~V~R~K▒~V~R~]▒~V~R▒~V~R~Z~D▒~V~R~V~G件
    eg▒~V~R~Z▒~V~R~K载▒~V~R~S▒~V~R~I~M▒~V~R~[▒~V~R▒~V~R~U▒~V~R~K▒~V~R~]▒~V~R▒~V~R~Z~D123.tar ▒~V~R~V~G件▒~V~R~Leg:http://localhost:5000/download?fileId=123.tar
    :return:
    """
    file_name = request.args.get('fileId')
    file_path = os.path.join(pwd,'',file_name)
    if os.path.isfile(file_path):
        return send_file(file_path,as_attachment=True)
    else:
        return "The downloaded file does not exist"


@app.route("/mxnet_json_file_download")
def mxnet_json_file_download():
    """
    ▒~V~R~K载src_file▒~V~R~[▒~V~R▒~V~R~U▒~V~R~K▒~V~R~]▒~V~R▒~V~R~Z~D▒~V~R~V~G件
    eg▒~V~R~Z▒~V~R~K载▒~V~R~S▒~V~R~I~M▒~V~R~[▒~V~R▒~V~R~U▒~V~R~K▒~V~R~]▒~V~R▒~V~R~Z~D123.tar ▒~V~R~V~G件▒~V~R~Leg:http://localhost:5000/download?fileId=123.tar
    :return:
    """
    file_name = request.args.get('fileId')
    file_path = os.path.join(pwd,'',file_name)
    if os.path.isfile(file_path):
        return send_file(file_path,as_attachment=True)
    else:
        return "The downloaded file does not exist"


@app.route("/pytorch_pth_file_download")
def pytorch_pth_file_download():
    """
    ▒~V~R~V~R~K载src_file▒~V~R~V~R~[▒~V~R~V~R▒~V~R~V~R~U▒~V~R~V~R~K▒~V~R~V~R~]▒~V~R~V~R▒~V~R~V~R~Z~D▒~V~R~V~R~V~G件
    eg▒~V~R~V~R~Z▒~V~R~V~R~K载▒~V~R~V~R~S▒~V~R~V~R~I~M▒~V~R~V~R~[▒~V~R~V~R▒~V~R~V~R~U▒~V~R~V~R~K▒~V~R~V~R~]▒~V~R~V~R▒~V~R~V~R~Z~D123.tar ▒~V~R~V~R~V~G件▒~V~R~V~R~Leg:http://localhost:5000/downll
oad?fileId=123.tar
    :return:
    """
    file_name = request.args.get('fileId')
    file_path = os.path.join(pwd,'pytorch',file_name)
    if os.path.isfile(file_path):
        return send_file(file_path,as_attachment=True)
    else:
        return "The downloaded file does not exist"


@app.route("/tensorflow_model_file_download")
def tensorflow_model_file_download():
    """
    ▒~V~R~V~R~V~R~K载src_file▒~V~R~V~R~V~R~[▒~V~R~V~R~V~R▒~V~R~V~R~V~R~U▒~V~R~V~R~V~R~K▒~V~R~V~R~V~R~]▒~V~R~V~R~V~R▒~V~R~V~R~V~R~Z~D▒~V~R~V~R~V~R~V~G件
    eg▒~V~R~V~R~V~R~Z▒~V~R~V~R~V~R~K载▒~V~R~V~R~V~R~S▒~V~R~V~R~V~R~I~M▒~V~R~V~R~V~R~[▒~V~R~V~R~V~R▒~V~R~V~R~V~R~U▒~V~R~V~R~V~R~K▒~V~R~V~R~V~R~]▒~V~R~V~R~V~R▒~V~R~V~R~V~R~Z~D123.tar ▒~V~R~V~R~V~RR
~V~G件▒~V~R~V~R~V~R~Leg:http://localhost:5000/downll
oad?fileId=123.tar
    :return:
    """
    file_name = request.args.get('fileId')
    file_path = os.path.join(pwd,'tensorflow',file_name)
    if os.path.isfile(file_path):
        return send_file(file_path,as_attachment=True)
    else:
        return "The downloaded file does not exist"


@app.route("/onnx_model_file_download")
def onnx_model_file_download():
    """
    ▒~V~R~V~R~V~R~V~R~K载src_file▒~V~R~V~R~V~R~V~R~[▒~V~R~V~R~V~R~V~R▒~V~R~V~R~V~R~V~R~U▒~V~R~V~R~V~R~V~R~K▒~V~R~V~R~V~R~V~R~]▒~V~R~V~R~V~R~V~R▒~V~R~V~R~V~R~V~R~Z~D▒~V~R~V~R~V~R~V~R~V~G件
    eg▒~V~R~V~R~V~R~V~R~Z▒~V~R~V~R~V~R~V~R~K载▒~V~R~V~R~V~R~V~R~S▒~V~R~V~R~V~R~V~R~I~M▒~V~R~V~R~V~R~V~R~[▒~V~R~V~R~V~R~V~R▒~V~R~V~R~V~R~V~R~U▒~V~R~V~R~V~R~V~R~K▒~V~R~V~R~V~R~V~R~]▒~V~R~V~R~V~R~V~R▒▒
~V~R~V~R~V~R~V~R~Z~D123.tar ▒~V~R~V~R~V~R~V~RR
~V~G件▒~V~R~V~R~V~R~V~R~Leg:http://localhost:5000/downll
oad?fileId=123.tar
    :return:
    """
    file_name = request.args.get('fileId')
    file_path = os.path.join(pwd,'onnx',file_name)
    if os.path.isfile(file_path):
        return send_file(file_path,as_attachment=True)
    else:
        return "The downloaded file does not exist"




# interface for set up server.
@app.route('/')
def index():
    print("Client connected!")
    return "Setup server successful!"

# interface for compute flops.
@app.route('/compiler/v1.0/flops')
def flops():
    return "flops compute complete!"

tasks = [
{       'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
},
{       'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
}
]

@app.route('/todo/api/v1.0/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})
    #return jsonify({'tasks': map(make_public_task, tasks)})


@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = list(filter(lambda t: t['id'] == task_id, tasks))
    if len(task) == 0:
        abort(404)
    return jsonify({'task': task[0]})

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/todo/api/v1.0/tasks', methods=['POST'])
def create_task():
    if not request.json or not 'title' in request.json:
        abort(400)
    task = {
        'id': tasks[-1]['id'] + 1,
        'title': request.json['title'],
        'description': request.json.get('description', ""),
        'done': False
    }
    tasks.append(task)
    return jsonify(({'task': task}), 201)

@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    task = list(filter(lambda t: t['id'] == task_id, tasks))
    if len(task) == 0:
        abort(404)
    if not request.json:
        abort(400)
    if 'title' in request.json and type(request.json['title']) != unicode:
        abort(400)
    if 'description' in request.json and type(request.json['description']) is not unicode:
        abort(400)
    if 'done' in request.json and type(request.json['done']) is not bool:
        abort(400)
    task[0]['title'] = request.json.get('title', task[0]['title'])
    task[0]['description'] = request.json.get('description', task[0]['description'])
    task[0]['done'] = request.json.get('done', task[0]['done'])
    return jsonify({'task': task[0]})

@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    task = filter(lambda t: t['id'] == task_id, tasks)
    if len(task) == 0:
        abort(404)
    tasks.remove(task[0])
    return jsonify({'result': True})


def make_public_task(task):
    new_task = {}
    for field in task:
        if field == 'id':
            new_task['uri'] = url_for('get_task', task_id=task['id'], _external=True)
        else:
            new_task[field] = task[field]
    return new_task


if __name__ == '__main__':
    #app.run(debug=True)
    #print('Server started, Listening at 5000...')
    app.run(host='0.0.0.0', port=5003, debug = True)
