import tqdm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import time
import torch

class Trainer(object):
	def __init__(self, model, loss_func, optimizer, metric_func=None, verbose=0, device='cuda'):
		self.model = model
		self.loss_func = loss_func
		self.optimizer = optimizer
		self.metric_func = metric_func
		self.verbose = verbose
		self.device = device

	def fit(self, train_loader, val_loader, epochs):
		losses = []
		running_loss = 0.0
		val_losses =[]
		val_loss = 0.0
		if self.metric_func is not None:
			metrics = []
			running_metric = 0.0
			val_metrics = []
			val_metric = 0.0
		else:
			metrics = None
			running_metric = None
			val_metric = None

		for epoch in range(epochs):
			starttime = time.time()

			running_loss, running_metric = self.train_step(train_loader, running_loss, running_metric)
			val_loss, val_metric = self.validation_step(val_loader, val_loss, val_metric)

			endtime = int(np.round(time.time() - starttime, decimals=0))
			
			try:
				its = np.round(len(train_loader) / endtime, decimals=2)
			except ZeroDivisionError:
				its = np.round(len(train_loader) / (endtime+1e-100), decimals=2)

			clear_output()
			losses.append(running_loss)
			val_losses.append(val_loss)
			if metrics is not None:
				metrics.append(running_metric)
				val_metrics.append(val_metric)

			if self.verbose==1:
				print(f'Epoch: {epoch} >>>>>>>>>>>>>> Loss: {running_loss} - Metric: {running_metric} --- Val Loss: {val_loss} - Val Metric {val_metric} ---------------- {endtime}s: {its}it/s')

			
			elif self.verbose==2:
				print(f'Epoch: {epoch} >>>>>>>>>>>>>> Loss: {running_loss} - Metric: {running_metric} --- Val Loss: {val_loss} - Val Metric {val_metric} ---------------- {endtime}s: {its}it/s')
				self.training_curves(epochs, losses, val_losses, metrics, val_metrics)
			
	def train_step(self, dataloader, running_loss, running_metric=None):
		for i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
			X_batch, Y_batch = data

			X_batch = X_batch.to(self.device)
			Y_batch = Y_batch.to(self.device)

			self.model.train()
			self.optimizer.zero_grad()
			outputs = self.model(X_batch)
			loss = self.loss_func(outputs, Y_batch)
			loss.backward()
			self.optimizer.step()
			running_loss += loss.item()

			if self.metric_func is not None:
				metric = self.metric_func(Y_batch.detach(), outputs.detach())
				running_metric += metric.item()

		running_loss /= len(dataloader)
		
		try:	
			running_metric /= len(dataloader)
		except:
			running_metric = None

		return running_loss, running_metric

	def validation_step(self, dataloader, running_loss, running_metric=None):
		for i, data in enumerate(dataloader):
			X_batch, Y_batch = data

			X_batch = X_batch.to(self.device)
			Y_batch = Y_batch.to(self.device)

			self.model.eval()
			with torch.no_grad():
				outputs = self.model(X_batch)
				loss = self.loss_func(outputs, Y_batch)
				running_loss += loss.item()

				if self.metric_func is not None:
					metric = self.metric_func(Y_batch.detach(), outputs.detach())
					running_metric += metric.item()

		running_loss /= len(dataloader)
		
		try:	
			running_metric /= len(dataloader)
		except:
			running_metric = None

		return running_loss, running_metric

	def evaluate(self, dataloader, to_device='cpu'):
		self.model.eval()
		with torch.no_grad():
			self.model = self.model.to(to_device)
			for i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
				X_batch, Y_batch = data
				if i == 0:
					inputs = X_batch
					predictions = self.model(X_batch)
					truths = Y_batch
				else:
					inputs = torch.cat([inputs, X_batch], axis=0)
					predictions = torch.cat([predictions, self.model(X_batch)], axis=0)
					truths = torch.cat([truths, Y_batch], axis=0)
		loss = self.loss_func(predictions, truths)
		metric = None
		if self.metric_func is not None:
			metric = self.metric_func(truths, predictions)
		print(f'Evaluation >>>>>>>>>>>>>>>> Loss: {loss} - Metric {metric}')
		return inputs, predictions, truths

	def predict(self, input, to_device='cpu'):
		self.model.eval()
		self.model = self.model.to(to_device)
		with torch.no_grad():
			output = self.model(input)
		return output

	@staticmethod
	def training_curves(iterations, losses, val_losses, metrics=None, val_metrics=None):
		if metrics is not None:
			fig, axes = plt.subplots(1,2, figsize=(12,4))
			fig.tight_layout(pad=3)
			axes[0].plot(losses, label='Train')
			axes[0].plot(val_losses, label='Val')
			axes[0].set_xlabel('Epoch')
			axes[0].set_ylabel('Loss')
			axes[0].set_ylim(bottom=0)
			axes[0].set_xlim([0, iterations])
			axes[0].legend()

			axes[1].plot(metrics, label='Train')
			axes[1].plot(val_metrics, label='Val')
			axes[1].set_xlabel('Epoch')
			axes[1].set_ylabel('Metric')
			axes[1].set_xlim([0, iterations])
			axes[0].legend()
			
			plt.show()
		else:
			fig = plt.figure(figsize=(6,4))
			plt.plot(losses, label='Train')
			plt.plot(val_losses, label='Val')
			plt.xlabel('Epoch')
			plt.ylabel('Loss')
			plt.xlim([0, iterations])
			plt.ylim(bottom=0)
			plt.legend()
			plt.show()


