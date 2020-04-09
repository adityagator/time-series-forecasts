import numpy as np
import pandas as pd

class Croston:
	def croston_method(total_data):
		extra_periods=12
		alpha=0.4
		d = np.array(total_data) 
		cols = len(d) 
		# Historical period length
		d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods
		
		#level (a), periodicity(p) and forecast (f)
		a,p,f = np.full((3,cols+extra_periods),np.nan)
		q = 1 #periods since last demand observation
		
		# Initialization
		first_occurence = np.argmax(d[:cols]>0)
		a[0] = d[first_occurence]
		p[0] = 1 + first_occurence
		f[0] = a[0]/p[0]
		# Create all the t+1 forecasts
		for t in range(0,cols):        
			if d[t] > 0:
				a[t+1] = alpha*d[t] + (1-alpha)*a[t] 
				p[t+1] = alpha*q + (1-alpha)*p[t]
				f[t+1] = a[t+1]/p[t+1]
				q = 1           
			else:
				a[t+1] = a[t]
				p[t+1] = p[t]
				f[t+1] = f[t]
				q += 1
		   
		# Future Forecast 
		a[cols+1:cols+extra_periods] = a[cols]
		p[cols+1:cols+extra_periods] = p[cols]
		f[cols+1:cols+extra_periods] = f[cols]
						  
		df = pd.DataFrame.from_dict({"Demand":d,"Forecast":f,"Period":p,"Level":a,"Error":d-f})
		return df["Forecast"].iloc[-1]

