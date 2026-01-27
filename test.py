from dotenv import load_dotenv                                                                                                                 
load_dotenv()                                                                                                                                  
                                                                                                                                                
from src.agent import BibleAgent                                                                                                               
agent = BibleAgent()                                                                                                                           
print('Agent created successfully')                                                                                                            
print(f'Loaded {len(agent.search.verses)} verses')                                                                                             
                                                                                                                                                
# Test a simple query                                                                                                                          
response = agent.chat('מי ברא את העולם?')                                                                                                      
print('\\n--- Response ---')                                                                                                                   
print(response)