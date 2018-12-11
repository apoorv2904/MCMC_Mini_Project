
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from collections import defaultdict
from scipy.stats import norm
import pickle as cPickle
import time
sns.set_style('white')
sns.set_context('talk')

def set_seed( seed ):
    np.random.seed(seed)
      

def plot_schedule( b_vector, title=""):
    """Plots temperature and beta schedule over time."""
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(14,5))
    ax1.plot(np.arange(len(b_vector)), b_vector )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Beta Value")
    ax1.set_title( "Beta Schedule")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Temperature")
    ax2.set_title( "Temperature Schedule")
    ax2.plot(np.arange(len(b_vector)), 1.0/b_vector )
    plt.suptitle(title)

def plot_error(e_vectors, x_axis_vector=None, title="", legend=[],
               xlabel="Time", fsave=' '):
    """Plots error over time (or the values in x_axis_vector). Inputs are
    lists of vectors to allow comparisons."""
    if x_axis_vector is None:
        x_axis_vector = np.arange(len(e_vectors))
    if not legend:
        legend = [" "] * len(e_vectors)
    plt.figure(figsize=(24,8))
    plt.hlines([0.1, 0.2, 0.3, 0.4, 0.5], x_axis_vector[0], x_axis_vector[-1],
               linestyles="--", color="lightgray")
    for i, e_vector in enumerate(e_vectors):
        e_vector_std = None
        if isinstance(e_vector, tuple):
            e_vector_std = e_vector[1]
            e_vector = e_vector[0]
        plt.errorbar(x_axis_vector, e_vector, yerr=e_vector_std, label=legend[i],
                    capthick=2, capsize=3)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Error")
    plt.ylim(0, 0.6)
    if len(fsave) > 0:
        plt.savefig(fsave + '.pdf')
    
def plot_energy_error(h_vectors, e_vectors, title="", legend=[]):
    """Plots energy and error over time. Inputs are lists of vectors to allow
    comparisons."""
    show_legend = True
    if not legend:
        legend = [" "] * len(h_vectors)
        show_legend = False
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(20,4))
    ax1.set_title(title)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Energy")
    ax1.set_yscale("log")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Error")
    ax2.set_ylim(0, 1)
    for y in [0.1, 0.2, 0.3, 0.4]:
        ax2.axhline(y, ls="--", color="lightgray")
    for i, (h_vector, e_vector) in enumerate(zip(h_vectors, e_vectors)):
        ax1.plot(np.arange(len(h_vector)), h_vector, label=legend[i])
        ax2.plot(np.arange(len(e_vector)), e_vector, label=legend[i])
    if show_legend:
        ax1.legend()
        ax2.legend()
      
    
def plot_acceptance_trend( a_vector, s_vector, title=""):
    """Plots acceptance probabilities and binary acceptance variable over time."""
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(20,4))
    ax1.plot(np.arange(len(s_vector)), s_vector,'.')
    ax1.set_title(title)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Acceptance Result")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Acceptance Probability")
    for y in [0.2, 0.4, 0.6, 0.8]:
        ax2.axhline(y, ls="--", color="lightgray")
    ax2.plot(np.arange(len(a_vector)), a_vector, '.')
    
    
def plot_beta_schedule( b_vector ):
    plt.figure(figsize=(10,4))
    plt.plot(np.arange(len(b_vector)),b_vector)
    plt.xlabel("Time")
    plt.ylabel("Beta")

    


def dump_alpha_comparison( exp_G, exp_M, fpickle ):
    save_object = defaultdict(defaultdict)
    save_object['exp_G'] = exp_G
    save_object['exp_M'] = exp_M
       
    
    file_handler = open("%s.pkl" %(fpickle),"wb")
    cPickle.dump(save_object,file_handler)
    file_handler.close()

def load_alpha_comparison(  fpickle ):
  
    file = open("%s.pkl" %(fpickle),"rb")
    save_object = cPickle.load(file)
    exp_G = save_object['exp_G'] 
    exp_M = save_object['exp_M'] 
    
    file.close()
    
    return exps_G, exps_M






import plotly.plotly as py
from plotly.graph_objs import Box, Figure
from plotly.offline import iplot
from plotly.graph_objs import Contours, Histogram2dContour, Marker, Scatter
import plotly.graph_objs as go
import plotly
print( plotly.__version__ )
import plotly.io as pio

def enable_plotly_in_cell():
    import IPython
    from plotly.offline import init_notebook_mode
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
    '''))
    init_notebook_mode(connected=False)
    
def dump_cooling_comparison( schedule_names, schedule_array, experiment_outputs, fpickle ):
    save_object = defaultdict(defaultdict)
    save_object['schedule_names'] = schedule_names
    save_object['schedule_array'] = schedule_array
    save_object['experiment_outputs'] = experiment_outputs    
    
    file_handler = open("%s.pkl" %(fpickle),"wb")
    cPickle.dump(save_object,file_handler)
    file_handler.close()

def load_cooling_comparison(  fpickle ):
  
    file = open("%s.pkl" %(fpickle),"rb")
    save_object = cPickle.load(file)
    schedule_names = save_object['schedule_names'] 
    schedule_array = save_object['schedule_array'] 
    experiment_outputs  = save_object['experiment_outputs']   
    file.close()
    
    return schedule_names, schedule_array, experiment_outputs


def violinplot_final_errors( schedule_names, experiment_outputs, title='', fsave='', x_axis='' ):
    n_exps = len( experiment_outputs[0].final_errors() )
    n_schedules = len( schedule_names )
    data = []
    for i in range(n_schedules):
        schedule_name = schedule_names[i]
        errors = experiment_outputs[i].final_errors()
        trace = {
                "type": 'violin',
                "x": [schedule_name] * n_exps,
                "y": errors,
                "name": schedule_name,
                "box": {
                    "visible": True
                },
                "meanline": {
                    "visible": True
                }
            }
        data.append(trace)
    
    
    layout = go.Layout( autosize=False, width=1200, height=800,
                       margin=go.layout.Margin( l=90, r=100, b=150, t=150, pad=4 ),
                       title=title, showlegend=True, 
                       font = dict( color = "black",
                                   size = 18),
                       yaxis=dict(title='Error'),
                       xaxis=dict(title=x_axis))
    fig = Figure(data, layout)
    iplot(fig, filename=fsave)
    if len(fsave) > 0:
        pio.write_image(fig, fsave + '.pdf')

def violinplot_min_energy_errors( schedule_names, experiment_outputs, title='', fsave='', x_axis='' ):
    n_exps = len( experiment_outputs[0].final_errors() )
    n_schedules = len( schedule_names )
    data = []
    for i in range(n_schedules):
        schedule_name = schedule_names[i]
        errors = experiment_outputs[i].min_energy_errors()
        trace = {
                "type": 'violin',
                "x": [schedule_name] * n_exps,
                "y": errors,
                "name": schedule_name,
                "box": {
                    "visible": True
                },
                "meanline": {
                    "visible": True
                }
            }
        data.append(trace)
    
    layout = go.Layout( autosize=False, width=1200, height=800,
                       margin=go.layout.Margin( l=90, r=100, b=150, t=150, pad=4 ),
                       title=title, showlegend=True, 
                       font = dict( color = "black",
                                   size = 18),
                       yaxis=dict(title='Error'),
                       xaxis=dict(title=x_axis))
    fig = Figure(data, layout)
    iplot(fig, filename=fsave)
    if len(fsave) > 0:
        pio.write_image(fig, fsave + '.pdf')

    
def boxplot_final_errors( schedule_names, experiment_outputs, title='', fsave='', x_axis='' ):
    n_exps = len( experiment_outputs[0].final_errors() )
    n_schedules = len( schedule_names )
    data = []
    for i in range(n_schedules):
        schedule_name = schedule_names[i]
        errors = experiment_outputs[i].final_errors()
        data.append( Box(y=errors, boxpoints=False, name=schedule_name) )
                    
    
    layout = go.Layout( autosize=False, width=1200, height=800,
                       margin=go.layout.Margin( l=90, r=100, b=150, t=150, pad=4 ),
                       title=title, showlegend=True, 
                       font = dict( color = "black",
                                   size = 18),
                       yaxis=dict(title='Error'),
                       xaxis=dict(title=x_axis))
    
    fig = Figure(data, layout)
    iplot(fig, filename=fsave)
    if len(fsave) > 0:
        pio.write_image(fig, fsave + '.pdf')


def boxplot_min_energy_errors( schedule_names, experiment_outputs, title='', fsave='', x_axis='' ):
    n_exps = len( experiment_outputs[0].final_errors() )
    n_schedules = len( schedule_names )
    data = []
    for i in range(n_schedules):
        schedule_name = schedule_names[i]
        errors = experiment_outputs[i].min_energy_errors()
        data.append( Box(y=errors, boxpoints=False, name=schedule_name))
        
    
    layout = go.Layout( autosize=False, width=1200, height=800,
                       margin=go.layout.Margin( l=90, r=100, b=150, t=150, pad=4 ),
                       title=title, showlegend=True, 
                       font = dict( color = "black",
                                   size = 18),
                       yaxis=dict(title='Error'),
                       xaxis=dict(title=x_axis))
    
    fig = Figure(data, layout)
    iplot(fig, filename=fsave)
    if len(fsave) > 0:
        pio.write_image(fig, fsave + '.pdf')



def violinplot_final_errors_sns( schedule_names, experiment_outputs, title='', fsave=None, x_axis='Schedule' ):
    n_exps = len( experiment_outputs[0].final_errors() )
    n_schedules = len( schedule_names )
    df = pd.DataFrame()
                    
    for i in range(n_schedules):
        schedule_name = schedule_names[i]
        errors = experiment_outputs[i].final_errors()
        df_temp = pd.DataFrame({ x_axis : [schedule_name] * n_exps, 'Error': errors })
        df = df.append(df_temp)
                    
    plt.figure(figsize=(20,12))
    # Usual boxplot
    g = sns.violinplot(x=x_axis, y='Error', data=df)#, palette="Pastel1")
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    
    plt.title( title )
    if fsave is not None:
        plt.savefig(fsave + '.pdf')

def violinplot_min_energy_errors_sns( schedule_names, experiment_outputs, title='', fsave=None, x_axis='Schedule' ):
    n_exps = len( experiment_outputs[0].final_errors() )
    n_schedules = len( schedule_names )
    df = pd.DataFrame()
                    
    for i in range(n_schedules):
        schedule_name = schedule_names[i]
        errors = experiment_outputs[i].min_energy_errors()
        df_temp = pd.DataFrame({ x_axis : [schedule_name] * n_exps, 'Error': errors })
        df = df.append(df_temp)
                    
    plt.figure(figsize=(20,12))
    # Usual boxplot
    g = sns.violinplot(x=x_axis, y='Error', data=df)#, palette="Pastel1")
    g.set_xticklabels(g.get_xticklabels(),rotation=45)

    plt.title( title )
    if fsave is not None:
        plt.savefig(fsave + '.pdf')
        
def boxplot_final_errors_sns( schedule_names, experiment_outputs, title='', fsave=None, x_axis='Schedule' ):
    n_exps = len( experiment_outputs[0].final_errors() )
    n_schedules = len( schedule_names )
    df = pd.DataFrame()
                    
    for i in range(n_schedules):
        schedule_name = schedule_names[i]
        errors = experiment_outputs[i].final_errors()
        df_temp = pd.DataFrame({ x_axis : [schedule_name] * n_exps, 'Error': errors })
        df = df.append(df_temp)
                    
    plt.figure(figsize=(20,12))
    # Usual boxplot
    g = sns.boxplot(x=x_axis, y='Error', data=df)#, palette="Pastel1")
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    plt.title( title )
    if fsave is not None:
        plt.savefig(fsave + '.pdf')
        
def boxplot_min_energy_errors_sns( schedule_names, experiment_outputs, title='', fsave=None, x_axis='Schedule' ):
    n_exps = len( experiment_outputs[0].final_errors() )
    n_schedules = len( schedule_names )
    df = pd.DataFrame()
                    
    for i in range(n_schedules):
        schedule_name = schedule_names[i]
        errors = experiment_outputs[i].min_energy_errors()
        df_temp = pd.DataFrame({ x_axis : [schedule_name] * n_exps, 'Error': errors })
        df = df.append(df_temp)
                    
    plt.figure(figsize=(20,12))
    # Usual boxplot
    g = sns.boxplot(x=x_axis, y='Error', data=df)#, palette="Pastel1")
    g.set_xticklabels(g.get_xticklabels(),rotation=45)

    plt.title( title )
    if fsave is not None:
        plt.savefig(fsave + '.pdf')

