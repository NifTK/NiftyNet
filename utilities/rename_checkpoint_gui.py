#source =('foo','bar/baz','bar/bing/barrow')
#target=('fib','cat/baz','cat/bing/barrow','asdf','sdfg','dfgh','fghj','ghjk','qwer','wert','erty','wewer','weert','errty')

import tensorflow as tf

from tkinter import ttk
from tkinter import filedialog
from tkinter import *
import csv
import utilities.rename_checkpoint_to_partial

class RenameGUI(object):
  def __init__(self,source=None,target=None,transform=None):
    self._target_variables=[]
    self._source_variables=[]
    self._transform_definition=[]
    self._source=None
    self.build_gui()
    if source:
      self.load_source(source)
    if target:
      self.load_target(target)
    if transform:
      self.load_transform(transform)
    self.reload_tables()
    
  def build_gui(self):
    self.root = Tk()
    frame=Frame(self.root)
    Grid.rowconfigure(self.root, 0, weight=1)
    Grid.columnconfigure(self.root, 0, weight=1)
    frame.grid(row=0, column=0, sticky='nsew')
    Grid.columnconfigure(frame,0,weight=1)
    Grid.columnconfigure(frame,2,weight=1)
    Grid.rowconfigure(frame,0,weight=1)
    Grid.rowconfigure(frame,1,weight=1)
    self.treeS = ttk.Treeview(frame)
    self.treeT = ttk.Treeview(frame)
    self.treeLinked = ttk.Treeview(frame)
    scrollS = ttk.Scrollbar(frame,orient="vertical",command=self.treeS.yview)
    scrollT = ttk.Scrollbar(frame,orient="vertical",command=self.treeT.yview)
    scrollLinked = ttk.Scrollbar(frame,orient="vertical",command=self.treeLinked.yview)
    self.treeS.grid(row=0,column=0,sticky='nsew')
    scrollS.grid(row=0,column=1,sticky='nsew')
    self.treeT.grid(row=0,column=2,sticky='nsew')
    scrollT.grid(row=0,column=3,sticky='nsew')
    self.treeLinked.grid(row=1,column=0,sticky='nsew',columnspan=3)
    scrollLinked.grid(row=1,column=3,sticky='nsew')
    buttonFrame = Frame(frame)
    buttonFrame.grid(row=2, column=0,columnspan=4, sticky='nsew')
    Grid.columnconfigure(buttonFrame, 0, weight=1)
    Grid.columnconfigure(buttonFrame, 1, weight=1)
    Grid.columnconfigure(buttonFrame, 2, weight=1)
    Grid.columnconfigure(buttonFrame, 3, weight=1)
    Grid.columnconfigure(buttonFrame, 4, weight=1)
    loadSourceButton = ttk.Button(buttonFrame,command=self.loadS,text='Load Source Checkpoint')
    loadSourceButton.grid(row=2,column=0,sticky='nsew')
    loadTargetButton = ttk.Button(buttonFrame,command=self.loadT,text='Load Target Checkpoint')
    loadTargetButton.grid(row=2,column=1,sticky='nsew')
    loadTransformButton = ttk.Button(buttonFrame,command=self.loadTfm,text='Load Transform')
    loadTransformButton.grid(row=2,column=2,sticky='nsew')
    saveTransformationButton = ttk.Button(buttonFrame,command=self.save,text='Save Transformation')
    saveTransformationButton.grid(row=2,column=3,sticky='nsew')
    saveCheckpointButton = ttk.Button(buttonFrame,command=self.saveC,text='Save transformed checkpoint')
    saveCheckpointButton.grid(row=2,column=4,sticky='nsew')
    self.treeS.bind("<Double-ButtonPress-1>",self.bDDown)
    self.treeT.bind("<Double-ButtonPress-1>",self.bDDown)
    self.treeLinked.bind("<Double-ButtonPress-1>",self.bDDownLinked)
  def run_gui(self):
    self.root.mainloop()
  def load_source(self,source):
    self._source=source
    self._source_variables=tf.contrib.framework.list_variables(source)
    self.reload_tables()
  def load_target(self,target):
    self._target_variables=tf.contrib.framework.list_variables(target)
    self.reload_tables()
  def load_transform(self,transform):
    with open(transform,newline='') as csvfile:
      r=csv.reader(csvfile)
      self._transform_definition=[row for row in r]
    if any(len(row)!=2 for row in self._transform_definition):
      raise ValueError('Each line must have a source and target variable name')
      return
    self.reload_tables()
  def spl(self,x):
    return [e+'/' for e in x.split('/')[:-1]]+[e for e in x.split('/')[-1:] if e]
  def clear_tree(self,tree):
    for ch in tree.get_children():
      tree.delete(ch)
  def reload_tables(self):
    self.clear_tree(self.treeS)
    self.clear_tree(self.treeT)
    self.clear_tree(self.treeLinked)
    if self._source_variables:
      self.add_to_tree(self.treeS,[self.spl(v) for v,s in self._source_variables])
    if self._target_variables:
      self.add_to_tree(self.treeT,[self.spl(v) for v,s in self._target_variables])
    [self.link(*r) for r in self._transform_definition]
  def add_to_tree(self,tree,definition):
    for parts in definition:
      for it in range(0,len(parts)):
        if not tree.exists(''.join(parts[:it+1])):
          tree.insert(''.join(parts[:it]),'end',''.join(parts[:it+1]),text=parts[it])

# events
# In top row, double-clicking when an item is selected creates a transformation link
# which moves the nodes down to the lower tree
# In bottom row, double clicking unlinks and moves variables back up to the top self.treeS
  def loadS(self):
    source = filedialog.askopenfilename(filetypes = (('Checkpoint index','*.index'),))
    if source.endswith('.index'):
       source=source[:-6]
    self.load_source(source)
  def loadT(self):
    target=filedialog.askopenfilename(filetypes = (('Checkpoint index','*.index'),))
    if target.endswith('.index'):
       target=target[:-6]
    self.load_target(target)
  def loadTfm(self):
    self.load_transform(filedialog.askopenfilename(filetypes = (('csv','*.csv'),)))
  def current_transform_as_variable_pairs(self):
    pairs=[]
    tfms = self.treeLinked.get_children('')
    for tfm in tfms:
      src,trg=tfm[:-1].split('>>')
      if src[-1]!='/':
        pairs.append((src,trg))
      else:
        ch=[c for c in self.get_family(self.treeLinked,tfm)[1:] if c[-1] != '/']
        for c in ch:
          pairs.append((c.replace(tfm,src,1),c.replace(tfm,trg,1)))
    return pairs
  def save(self):
    filename= filedialog.asksaveasfilename(defaultextension=".csv", filetypes = ( ("csv file", "*.csv"), ) )
    with open(filename,mode='w',newline='') as file:
      if file is None:
          return
      w=csv.writer(file)
      [w.writerow(r) for r in self.current_transform_as_variable_pairs()]

  def saveC(self):
    if not self._source:
      raise ValueError('Source must be set before saving checkpoint')
    filename = filedialog.asksaveasfilename(defaultextension=".partial", filetypes = ( ("partial checkpoint file", "*.partial"), ) )
    if filename:
      utilities.rename_checkpoint_to_partial.rename_checkpoint_to_partial(self._source,filename,self.current_transform_as_variable_pairs())

  def get_family(self,tree,id):
    return [id]+[j for i in tree.get_children(id) for j in self.get_family(tree,i)]

  def clear_if_empty(self,tree,id):
    if len(self.get_family(tree,id))==1:
      parent=tree.parent(id)
      tree.delete(id)
      if parent:
        self.clear_if_empty(tree,parent)
  
  def link(self,src,trg):
    if src[-1]=='/' and trg[-1]=='/': # link scopes
      sCh=self.get_family(self.treeS,src)[1:]
      tCh=self.get_family(self.treeT,trg)[1:]
      matches=set([c.replace(src,'',1) for c in sCh if c[-1]!='/']).intersection(set([c.replace(trg,'',1) for c in tCh if c[-1]!='/']))
      self.add_to_tree(self.treeLinked,[[src+'>>'+trg+';']+self.spl(c) for c in matches])
      for c in matches:
        self.clear_if_empty(self.treeS,src+c)
        self.clear_if_empty(self.treeT,trg+c)
    elif src[-1]!='/' and trg[-1]!='/': # link variables
      self.add_to_tree(self.treeLinked,[[src+'>>'+trg+';']])
      self.clear_if_empty(self.treeS,src)
      self.clear_if_empty(self.treeT,trg)
    
  def bDDown(self,event):
    idS=self.treeS.focus()
    idT=self.treeT.focus()
    if not idS or not idT:
      return
    self.link(idS,idT)

  def bDDownLinked(self,event):
    id=self.treeLinked.focus()
    tfm,path=id.split(';',1)
    src,trg=tfm.split('>>')
    nodesToUnlink=self.get_family(self.treeLinked,id)
    addToTree(self.treeS,[self.spl(i.replace(tfm+';',src,1)) for i in nodesToUnlink if i[-1]!='/'])
    addToTree(self.treeT,[self.spl(i.replace(tfm+';',trg,1)) for i in nodesToUnlink if i[-1]!='/'])
    self.treeLinked.delete(id)
def main(argv):
  gui=RenameGUI(*argv)
  gui.run_gui()
  
if __name__=='__main__':
  main(sys.argv[1:])

