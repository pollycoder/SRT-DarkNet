from selenium.webdriver.common.keys import Keys
from torutils import TorBrowserDriver
import sys
sys.path.append('../..')
from helper import utils, log
from xvfbwrapper import Xvfb
import os
from dumputils import Sniffer
import time

class Visit(object):
    """Hold info about a particular visit to a page."""

    def __init__(self, instance_num, page_urls, comb_dir, tor_controller, 
                 num_tabs, tbb_path, xvfb, screenshot):
        # load
        self.instance_num = instance_num
        self.page_urls = page_urls
        self.comb_dir = comb_dir
        self.tor_controller = tor_controller
        self.tbb_path = tbb_path
        self.visit_dir = None
        self.visit_log_dir = None
        self.xvfb = xvfb
        self.screenshot = screenshot
        
        # init visit dir
        self.init_visit_dir()
        self.pcap_path = os.path.join(self.visit_dir, "tcp.pcap")
        
        # use xvfb
        if self.xvfb:
            self.xvfb_display = utils.start_xvfb()
        
        # Create new instance of TorBrowser driver
        self.tb_driver = TorBrowserDriver(
            tbb_logfile_path=os.path.join(self.visit_dir, "logs", "firefox.log"), tbb_path=self.tbb_path)

        self.sniffer = Sniffer()  # sniffer to capture the network traffic

    def init_visit_dir(self):
        """Create results and logs directories for this visit."""
        self.visit_dir = os.path.join(self.comb_dir, 'inst-' + str(self.instance_num))
        utils.create_dir(self.visit_dir)
        self.visit_log_dir = os.path.join(self.visit_dir, 'logs')
        utils.create_dir(self.visit_log_dir)

    def cleanup_visit(self):
        """Kill sniffer and Tor browser if they're running."""
        print("INFO\tCleaning up visit.")
        print("INFO\tCancelling timeout")
        utils.cancel_timeout()

        if self.sniffer and self.sniffer.is_recording:
            print("INFO\tStopping sniffer...")
            self.sniffer.stop_capture()
        if self.tb_driver and self.tb_driver.is_running:
            # shutil.rmtree(self.tb_driver.prof_dir_path)
            print("INFO\tQuitting selenium driver...")
            self.tb_driver.quit()

        # close all open streams to prevent pollution
        print("INFO\tClose all open streams")
        self.tor_controller.close_all_streams()
        if self.xvfb:
            print("INFO\tStop xvfb")
            utils.stop_xvfb(self.xvfb_display)

    def take_screenshot(self, index_url):
        try:
            out_png = os.path.join(self.visit_dir, '{}.png'.format(index_url))
            print("INFO\tTaking screenshot of %s to %s" % (self.page_urls, out_png))
            self.tb_driver.get_screenshot_as_file(out_png)
        except:
            print("ERROE\tException while taking screenshot of: %s" % self.page_urls)
            return False
        return True

    def get(self):
        """Call the specific visit function depending on the experiment."""

        utils.timeout(utils.HARD_VISIT_TIMEOUT)
      
        print('INFO\tcapture start in {}'.format(utils.cal_now_time()))
        self.sniffer.start_capture(
            self.pcap_path,
            'tcp and not host %s and not host %s and not port 22'
            % (utils.USED_GATEWAY_IP, utils.LOCALHOST_IP))

        try:
            self.tb_driver.set_page_load_timeout(utils.SOFT_VISIT_TIMEOUT)
        except:
            print("INFO\tException setting a timeout {}".format(self.page_urls))

        time.sleep(utils.WAIT_AFTER_DUMP)

        for page_url in self.page_urls:
            newTab = 'window.open("https://%s");' % page_url
            print('INFO\tCrawling URL: {} in {}'. format(page_url, utils.cal_now_time()))
            self.tb_driver.execute_script(newTab)
            self.tb_driver.switch_to.window(self.tb_driver.window_handles[-1])
            time.sleep(utils.INTERVAL_BETWEEN_TABS)
        time.sleep(utils.WAIT_FOR_COMB)
        print('INFO\tEnd crawling urls in {}'.format(utils.cal_now_time()))
        
        if self.screenshot:
            for index in range(len(self.page_urls)):
                self.tb_driver.switch_to.window(self.tb_driver.window_handles[index+1])
                if self.take_screenshot(index) is False:
                    break
        self.cleanup_visit()


if __name__ == "__main__":
    print('test ok!')
    