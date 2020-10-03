from setuptools import setup, find_packages
import gym_epidemic
setup(
  name = 'gym_epidemic',        
  packages = ['gym_epidemic'],   
  version = '0.0.2b',
  license='BSD-3',  
  description = 'An OpenAI Gym to benchmark AI Reinforcement Learning algorithms in epidemic control problems',  
  author = 'Marcus Lapeyrolerie',                 
  author_email = 'marcuslapeyrolerie@me.com',      
  url = 'https://github.com/boettiger-lab/gym_epidemic',   
  #download_url = 'https://github.com/boettiger-lab/gym_epidemic/releases/tag/v0.0.3',
  keywords = ['Reinforcement Learning', 'Epidemic Control', 'Epidemics', 
              "COVID-19", "AI", "stable-baselines", "OpenAI Gym", 
              "Artificial Intelligence", "Epidemiology"],
  install_requires=[ 
          'gym',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',    
    'Intended Audience :: Developers', 
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
