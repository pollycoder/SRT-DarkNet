import sys
import os
sys.path.append('../..')
from helper import utils, log, closed_world
from torutils import TorController
from visit import Visit
from selenium.common.exceptions import TimeoutException
import numpy as np
import time
import traceback
import json


class Crawler():
    '''
    Provides methods to collect traffic traces.
    '''
    # Crawler(torrc_dict, urls_100_list, urls_5w_list, open_world, num_tabs, tbb_path, output, xvfb, screenshot)
    def __init__(self, torrc_dict, urls_df, open_world, tbb_path, output, xvfb=False, screenshot=False):
        # Create instance of Tor controller and sniffer used for the crawler
        self.crawl_dir = None
        self.crawl_logs_dir = None
        self.visit = None
        self.log_file = None
        self.tor_log = None
        self.urls_df = urls_df 
        self.tbb_path = tbb_path
        self.xvfb = xvfb
        self.screenshot = screenshot
        self.open_world = open_world
        self.debug_dir = '/root/debug.json'

        # Initializes 
        self.init_crawl_dirs(output)  
        self.tor_controller = TorController(torrc_dict, tbb_path, self.tor_log)
        self.tor_process = None
        self.tb_driver = None
    
    def init_crawl_dirs(self, output):
        #Creates results and logs directories for this crawl.
        self.crawl_dir= self.create_crawl_dir(output)
        #self.log_file = os.path.join(self.crawl_logs_dir, "crawl.log")
        #self.tor_log = os.path.join(self.crawl_logs_dir, "tor.log")

    def create_crawl_dir(self, output):
        # Create a timestamped crawl.
        if not os.path.exists(output):
            crawl_dir=utils.create_dir(output)  # ensure that we've a results dir
        # 'output/crawl'
        
        # 'output/crawl+timestamped/logs'
        #crawl_logs_dir = os.path.join(crawl_dir, 'logs')
        #utils.create_dir(crawl_logs_dir)
        return crawl_dir

    def write_bugs(self,bug_text,bug_urls):
        
        bugger={'urls':bug_urls,'text':bug_text}
        with open(self.debug_dir,"a") as f:
            json.dump(bugger,f)
            
    
    def crawl(self,num_instances=10):
        # for each batch
        #urls_df是dataframe形式，一行是一个comb
        combs=self.urls_df
        print("INFO\tCrawl configuration: instances: {0}, num of combinations: {1}, crawl dir: {2}".format\
            (num_instances, len(combs), self.crawl_dir))
        
        site_num = 0
        
        print("INFO\tRestarting Tor in {}".format(utils.cal_now_time()))
        self.tor_controller.restart_tor()
        sites_crawled_with_same_proc = 0

        for index, row in combs.iterrows():
            #计算标签数
            print("index=",index)
            page_urls=row.to_list()
            n_tabs=0
            real_urls=[]
            print("page_urls",page_urls)
            for i in range(len(page_urls)):
                if type(page_urls[i])!=float:
                    cur_url=page_urls[i]
                    cur_url=cur_url.replace('https://www.', '')
                    cur_url=cur_url.replace('https://', '')
                    n_tabs+=1
                    
                    real_urls.append(cur_url)
            
            
            print("INFO\tNumber of tabs = {}".format(n_tabs))


            print('INFO\tCrawling combination No.{}: {} in {}'.format(index, real_urls, utils.cal_now_time()))
            comb_dir = utils.create_dir(os.path.join(self.crawl_dir,'comb-'+str(index)))
            sites_crawled_with_same_proc += n_tabs

            if sites_crawled_with_same_proc > utils.MAX_SITES_PER_TOR_PROCESS:
                print("INFO\tRestarting Tor in {}".format(utils.cal_now_time()))
                self.tor_controller.restart_tor()
                sites_crawled_with_same_proc = 0

            with open(os.path.join(comb_dir, 'label'), 'w') as fp:
                for now_url in real_urls:
                    fp.write(now_url + ' ')
                fp.write('\n')

            for instance_num in range(num_instances):
                print('INFO\nCrawling {} of {} instances in {}'.format(real_urls, instance_num, utils.cal_now_time()))

                self.visit = None
                try:
                    self.visit = Visit(instance_num, real_urls, comb_dir, 
                                        self.tor_controller, n_tabs, self.tbb_path, self.xvfb, self.screenshot)
                    self.visit.get()
                except KeyboardInterrupt:  # CTRL + C
                    raise KeyboardInterrupt
                except TimeoutException as exc:
                    print("CRITICAL\tVisit timed out! %s %s" % (exc, type(exc)))
                    bug_text=traceback.format_exc()
                    self.write_bugs(bug_text,real_urls)
                    #print(traceback.format_exc())
                    if self.visit:
                        self.visit.cleanup_visit()
                except Exception as exc:
                    print("CRITICAL\tException crawling: %s" % exc)
                    bug_text=traceback.format_exc()
                    self.write_bugs(bug_text,real_urls)
                    if self.visit:
                        self.visit.cleanup_visit()
            # END - for each visit
            site_num += 1
            time.sleep(utils.INTERVAL_BETWEEN_COMBS)
        
    def stop_crawl(self, pack_results=True):
        """ Cleans up crawl and kills tor process in case it's running."""
        print("Stopping crawl...")
        if self.visit:
            self.visit.cleanup_visit()
        self.tor_controller.kill_tor_proc()
    
if __name__ == "__main__":
    print('test')
                