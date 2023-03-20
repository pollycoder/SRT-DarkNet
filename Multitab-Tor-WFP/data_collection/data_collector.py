import argparse
import traceback
import os
import sys
sys.path.append('..')
from helper import utils, log
sys.path.append('models')
from models.crawler import Crawler

import pandas as pd

if __name__ == '__main__':
    open_world = True
    # Init
    SOCKS_PORT = utils.USED_SOCKS_PORT
    CONTROLLER_PORT = utils.USED_CONTROL_PORT
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Crawl a list of URLs in several batches.')
    
    # Add arguments
    #parser.add_argument('--ntabs', default=3, type=int, help='Number of tabs in the tor browser')
    #parser.add_argument('--batch', default=5, type=int, help='Number of batches')
    #parser.add_argument('--comb', default=4, type=int, help='Number of combinations')
    parser.add_argument('--instance', default=10, type=int, help='Number of instances')
    parser.add_argument('--output', default='output', type=str, help='Path of the output file')
    parser.add_argument('--tbbpath', default='../tbb/tor-browser_zh-CN', type=str, help='Path of tbb')  
    parser.add_argument('--start_comb', default=1, type=int, help='服务器开始爬的comb行数')
    parser.add_argument('--end_comb', default=2501, type=int, help='服务器停止爬的comb行数')
    parser.add_argument('--xvfb', default=False, type=bool, help='Use XVFB (for headless testing)') 
    parser.add_argument('--screenshot', default=False, type=bool, help='Capture page screenshots)')
    parser.add_argument('--urls_check', default='./input/webpage_2-5tab.pickle', type=str, help='Alexa Top 1w URLs file path')
    
    args = parser.parse_args()

    # Load data
    # num_tabs = args.ntabs
    # num_batches = args.batch
    # num_comb = args.comb
    
    num_instances = args.instance
    output = args.output
    tbb_path = args.tbbpath
    xvfb = args.xvfb
    screenshot = args.screenshot
    urls_check_path = args.urls_check
    start_comb=args.start_comb-10000
    end_comb=args.end_comb-10000

    assert os.path.isfile(urls_check_path)
    assert urls_check_path
   
    whole_list = pd.read_pickle(urls_check_path)
        #这是dataframe类型
    
    #10000开始，2-5tab的时候减去10000
    urls_df=whole_list.iloc[start_comb:end_comb+1]
    
    #0-22832,0-4
    #一行就是一comb
    torrc_dict = {'SocksPort': str(SOCKS_PORT),
             'ControlPort': str(CONTROLLER_PORT)}
    
    crawler = Crawler(torrc_dict, urls_df, open_world,tbb_path, output, xvfb, screenshot)
    print('INFO\tInit crawler finish in {}'.format(utils.cal_now_time()))
    print("INFO\tCommand line parameters: %s" % sys.argv)
    
    # Run the crawl
    
    try:
        crawler.crawl(num_instances)
    except KeyboardInterrupt:
        log.wl_log.warning("WARNING\tKeyboard interrupt! Quitting in {}...".format(utils.cal_now_time()))
    except Exception as e:
        log.wl_log.error("ERROR\tException in {}: \n {}".format(utils.cal_now_time(), traceback.format_exc()))
    finally:
        crawler.stop_crawl()
    
    print('INFO\tData Collection done in {}'.format(utils.cal_now_time()))