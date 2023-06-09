# Copyright 2022 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 2023/05/05
# EpochChangeCallback.py

# encodig: utf-8

import os
import shutil
import traceback


import tensorflow as tf


class EpochChangeCallback(tf.keras.callbacks.Callback):

  ##
  # Constructor
  def __init__(self, eval_dir):
    self.eval_dir = eval_dir
    if os.path.exists(self.eval_dir):
      shutil.rmtree(self.eval_dir)

    if not os.path.exists(self.eval_dir):
      os.makedirs(self.eval_dir)
    self.train_losses_file     = os.path.join(self.eval_dir, "train_losses.csv")  
    self.train_accuracies_file = os.path.join(self.eval_dir, "train_accuracies.csv")  
    try:
      if not os.path.exists(self.train_losses_file):
        with open(self.train_losses_file, "w") as f:
          header = "epoch, loss, val_loss\n"
          f.write(header)
    except Exception as ex:
        traceback.print_exc()

    try:
      if not os.path.exists(self.train_accuracies_file):
        with open(self.train_accuracies_file, "w") as f:
          header = "epoch, accuracy, val_accuracy\n"
          f.write(header)
    except Exception as ex:
        traceback.print_exc()


  def on_epoch_end(self, epoch, logs):
    #print("\n   on_epoch_end :epoch:{}".format(epoch))
    
    acc     = 0
    if 'accuracy' in logs:
      acc      = logs.get('accuracy')
    elif 'acc' in logs:
      acc      = logs.get('acc')

    val_acc = 0
    if 'val_accuracy' in logs:
      val_acc  = logs.get('val_accuracy')
    elif 'val_acc' in logs:
      val_acc  = logs.get('val_acc')

    loss     = logs.get('loss')
    val_loss = logs.get('val_loss')
   
    NL  = "\n"

    try:
       with open(self.train_losses_file, "a") as f:
         losses    = "{}, {:.4f}, {:.4f}".format(epoch, loss, val_loss)
         f.write(losses + NL)
    except Exception as ex:
        traceback.print_exc()

    try:
       with open(self.train_accuracies_file, "a") as f:
         accuraies = "{}, {:.4f}, {:.4f}".format(epoch, acc,  val_acc)
         f.write(accuraies + NL)
 
    except Exception as ex:
        traceback.print_exc()
