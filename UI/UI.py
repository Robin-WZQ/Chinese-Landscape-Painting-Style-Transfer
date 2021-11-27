import tkinter
import tkinter.filedialog
from PIL import Image,ImageTk
from torchvision import transforms as transforms
from test import main,model


# 创建UI
win = tkinter.Tk()
win.title("picture process")
win.geometry("1280x1080")

# 声明全局变量
original = Image.new('RGB', (300, 400))
save_img = Image.new('RGB', (300, 400))
count = 0
e2 = None
e2 = str(e2)
file_name = None
img2 = tkinter.Label(win)


def choose_file():
	'''选择一张照片'''
	select_file = tkinter.filedialog.askopenfilename(title='select the picture')
	global file_name
	file_name=select_file
	e.set(select_file)
	load = Image.open(select_file)
	load = transforms.Resize((400,400))(load)
	# 声明全局变量
	global original
	original = load
	render = ImageTk.PhotoImage(load)
	img  = tkinter.Label(win,image=render)
	img.image = render
	img.place(x=100,y=100)

def coloring():
    '''图片生成'''
    model()
    new_img = Image.open('generate.png')
    new_img = transforms.Resize((400,400))(new_img)
    render = ImageTk.PhotoImage(new_img)

    global img2
    img2.destroy()
    img2  = tkinter.Label(win,image=render)
    img2.image = render
    img2.place(x=800,y=100)

def transfer():
    main(file_name)
    model()
    new_img = Image.open('generate.png')
    new_img = transforms.Resize((400,400))(new_img)
    render = ImageTk.PhotoImage(new_img)

    global img2
    img2.destroy()
    img2  = tkinter.Label(win,image=render)
    img2.image = render
    img2.place(x=800,y=100)

def edge_detect():
    '''边缘检测'''
    main(file_name)
    new_img = Image.open('canny&HED.jpg')
    new_img = transforms.Resize((400,400))(new_img)
    render = ImageTk.PhotoImage(new_img)

    global img2
    img2.destroy()
    img2  = tkinter.Label(win,image=render)
    img2.image = render
    img2.place(x=800,y=100)



e = tkinter.StringVar()
e_entry = tkinter.Entry(win, width=68, textvariable=e)
e_entry.pack()

# 文件选择
button1 = tkinter.Button(win, text ="Select", command = choose_file)
button1.pack()

button2 = tkinter.Button(win, text="edge detect" , command = edge_detect,width=20,height =1)
button2.place(x=570,y=200)

button3 = tkinter.Button(win, text="coloring" , command = coloring,width=20,height =1)
button3.place(x=570,y=300)

button4 = tkinter.Button(win, text="style transfer" , command = transfer,width=20,height =1)
button4.place(x=570,y=400)

label1 = tkinter.Label(win,text="Original Picture")
label1.place(x=250,y=50)

label2 = tkinter.Label(win,text="style transfer!")
label2.place(x=950,y=50)

# 退出按钮
button0 = tkinter.Button(win,text="Exit",command=win.quit,width=20,height =1)
button0.place(x=570,y=650)
win.mainloop()
