from collections import deque
from collections import namedtuple
import os
import re

class Learner(object):
  def __init__(self, model, model_io_mgr=None, save_interval=100):
    self.model = model
    self.model_io_mgr = model_io_mgr
    self.n_iter = 0
    self.save_interval = save_interval
    if model_io_mgr:
      self.n_iter = model_io_mgr.load()

  def train(self):
    self.do_train()
    self.n_iter += 1
    if self.n_iter % save_interval == 0:
      self.model_io_mgr.save(self.n_iter)

class ModelFileManager(object):
  def __init__(self, model, name, n_history=5, folder="."):
    self.model = model
    self.n_history = n_history
    self.folder = folder
    self.name = name
    self.file_pattern = r"^.*{}.hdf5.(\d+)$".format(name)
    self.reload_savepoints()

  def reload_savepoints(self):
    savepoints = []
    for name in os.listdir(self.folder):
      m = re.match(self.file_pattern, name)
      if not m:
        continue
      self.savepoints.append((os.path.abspath(name), int(m.group(1))))
    self.savepoints = deque(sorted(savepoints, key=lambda x: x[1]))

  def load(self, file=None):
    if not file:
      try:
        file = self.savepoints[-1][0]
      except:
        return
    self.model.load_weights(file)
    return int(re.match(self.file_pattern, file).group(1))

  def save(self, n_iter):
    file = os.path.join(self.folder, "{}.hdf5.{}".format(self.name, n_iter))
    if len(self.savepoints) == self.n_history:
      os.remove(self.savepoints.popleft()[0])
    self.savepoints.append((file, n_iter))
    self.model.save_weights(file)
