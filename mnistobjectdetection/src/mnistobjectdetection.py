# %% [markdown]
# # <b> MNIST Object Detection CNNs
# 

# %% [markdown]
# ## **Context**

# %% [markdown]
# The MNIST handwritten digit classification problem is a standard and widely used dataset in computer vision and deep learning.
# Although the dataset is effectively solved, it can be used as the basis for learning and practicing how to develop, evaluate, and use convolutional deep learning neural networks for image classification and object detection from scratch. This includes developing a robust test harness for estimating model performance and exploring improvements to the model.
# 
# The MNIST dataset is essentially a dataset of 10,000 small square 28Ã—28 pixel grayscale images of handwritten single digits between 0 and 9.  <br><br>
# <b>MNIST Object Detection Data: Overview </b>
# - This dataset is provided in a zipped format. 
# - This zip consists of two folders - train and test. Both these folders consist of two more sub-folders, images and labels. 
# - The images subfolder in the train folder consists of images starting from **0.png** to **9000.png** whereas the labels folder consists of text files starting from **0.txt** to **9000.txt**. Similarly, the test folder consists of images starting from **0.png** to **1000.png** whereas the labels folder consists of text files starting from **0.txt** to **1000.txt**. <br>
# 
# ![mnist_data.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA9EAAACICAYAAADgWWjRAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAFiUAABYlAUlSJPAAACv6SURBVHhe7d1vaBzpneDxn5c98MK+mIE5aIEc0sGBtdmBkeFALXZerAYfREMC00ID0yIvZpUNJJ7si7FnYE6yX8itLEw8c5C152AZJ5Cj2zBGGrjDDqxx54WD2lwWaWAWO7DDKEQGNWzAhgnYsIG656l6quqp6iqpqlUtdbe+H3ik7urq+vvU89Svq56njjmKWLYfPXL/nxgfd/8DRSJ/oZ/IX8DRw3GPfiJ/AfDZ5cGfua8AAAAAAMCeCKIBAAAAAMiIIBoAAAAAgIyO/X57O9Im+msnTphXAAAAAADAp+JnrkQDAAAAAJBVV+/cx44dc//HBgOFIH8BAIpEvQIAOAh2fcOVaAAAAAAAMiKIBgAAAAAgI4JoAAAAAAAyIogGAAAAACAjgmgAAAAAADIiiAYAAAAAICOCaAAAAAAAMiKIBgAAAAAgI4JoAKPt/or7cPxsaUXa5mv79qgps/50Lxc2VaMtK5HlTkhFzlNtw9kbHfMGAJDqsOqcRKqueL0pe5fe1ClAXgTRADCKLk2pE5/9n6C1L6uTp8qSeQcAGApuMD8lhZXe1ClABEE0APTDeE1WHUccnS5WzMCDtiRTma5CpFAnYVOXzGsAwJBoy0pfAlXqFMBHEA1gtE0ueoFskNalbj6SuYbsRD5blMMKd3vWtQ4qtYM1FLk5L9fvm9cAgP4a9jqHOgXIhCAaAHxWW7bZG21pvm7agqm04p802G2dgzQrzUfmc19Km2j3VjbrO+F7LwXz2Q99Emed9Czdid2Al2Ed4rfcrdXG3PEi7dis7RWmfrfxA4BR0onUNW5KaX/cuTEbHc9NsTI3fhu3CnrH9Hj7uYK8V52SoS6gTsGoIYgGgARrtSmZv2neSF2mJ9U/XcGfmJc1b6BlTeZP5A2Ave/Eb21bqiQE5L2YnA6vfny+FZ48FbQO7slc4u2CSzKV9KMCACBGd+g1ZtU1hm5/HAt6dRA6Vusuub0y9wACzZQ6pai6gDoFw4YgGgASVaWx7d/Opm+560jzQ1PB27e77fbr/J7qsm6ms75sBqlgdvVez9cLLGUpz5mXN7dky32RfR0qF6PDq80dd9zVN0rqXVuu+ydzy+veNFTaaVa9YYWtAwCMrvZl/4pxWBcE5e7NeTkXXKVtS8v/wTWx7FaBpn/12r2dPOEW8k9qokvv3iXVKdnrAuoUjBqCaABIsnxBauPmtasktU+8ij1yMjJeVuF2b+rtsD1c5c1GMJ21L7zTk+IVtQ4VWTQnOXanaaXyGfMKALCrR025YgLjanMhbButgmD/R9W1T1veFd9HW7LhDlFurkrLvyprt78+lA4si6oLqFMwfAiiASBB9WTZvEpj2rEl3hqdRVXKkSD9MOx3HXzmGaOJt+IBALqowNgvd/32wX4Kmvn4V3zHp2XWvwqsvqWb3vjjDtbzlouqC6hTMPgIogEgq0iHXAnt2AaKOvlKWr6i1iHSAUyBzyIFABgbsuVeddZ3EVm3aFv8ALyQTil3lVKnFFUXUKdgyBBEA0AmbVmxr9j67cy2w9uwB8r9VngSsjxtbhUsaB10IG5fIfDbsFnt3QAA2fjtg7vTqtWsKLzlOexDI7RU6XPnYkl1SlF1AXUKhhBBNABkYZ1AuCc8++6kpY9iJyT1s6aNWUHr0Lm3GgTi9bY60TmUtngAMMSsvijy9oPhdtLlBtQ70ghu8/avWvdBSp1SVF1AnYJhRBANADmFJzwdab693/bE++Q/A9ROsavNC/rxXDFFrcPGVth77Ip9JQEAkM5u53xpKrwdWwesflnu97ht3+oceYa0fYv1mWL62eixTimqLqBOwbAgiAaALOxnZOpneLonF7E2xfbzmAdCXdbtq81518G+UmLa3elObEovz3YNj7dh618P4wAwCkpS+zBsSrNUiQesVWm8aa7IWj12h2V3tNy1n/YQfRyVCYpjz53uTbROyV0XUKdghBBEA0Amuj1avGMX71nSwbMs7UePHDKvjZ19UqXlXIfxmqzG2qS5JzJ6eFc7au85p+GJXqu/7fMAYNjpsjRyS7bh9ldht4c2t3AnthH2yvDFyNXhhI7Igmc79yaxTslbF1CnYIQcc3SjCov+9UeLDR5c+rYX+zYTQ7epiBYoB6tzY1aul1cPfBkOa75ZDV3+AgAMNOoVAMBBsOubIb4SvfvzTd3bYgq5dSUvb7nGar22MOzVYc0XAAAAAI6OoQ2i25czPN9UtwOJdMDQf50b5w7l2bGHNV8AAAAAOEqGM4i+vyJTl8xrxe0O3wlT5Pl5l65Ic0DaKAIAAAAAhttQBtHtO2F/fUltn3XnC24gndAxg699Wff8Z6fZxGA7HM/7PP694JEECbdT+z0thuMo9mMK3HRA8wUAAAAA7NsQBtFtaflXoVOeVae5vRjaj3YJtGVFBZj2lWzPmsyf8LraT+Z9Hv/eUiU5CE6iO/061vXMO2+66QHv/ucLAAAAACjI77e3HTupQW4aXOtO3SyjLK+bYdmtL5vvqlRvm4HOjtOY84dXnca2GazY44vU1dy7h1ebO2aomlKzGgwPp69sN5xq1/j2fMNpa4XNd8D4y0gikUgkEolEIpFIw5Z0zDx8V6IfbcmGeZnbo6Zc8a/oLq9bt4HbD7xfk/mfJ3dGZj/IvvJm+Dy7LA+Ab//c70W8LhfeCB5Tb813SVopV6P3M18AAAAAQHH+7MT4uNhplHXurZpAVgWmZyOPixcZn5ZZ/2H3iQ90r0q5583Tka3PzUsVLE9ZbZvtR3RtbCXdSr6f+Q4ufVGaRCo6+ZI+I5FIo5l8SZ+RSPtNvqTPSCTS0Uo+HTMP35Xo8bKcMS+Tg93hxZVlAAAAABhsQ9ixWEWmg0dYpd8C7fVmvTKAQXZd1hN+2XDTxdjVcQAAAADAQBnKR1xVztbNK91LdXfP1roXbK83a+/Waf/z0suzQXvipTux8PpRS1ZvmtfL00Eb5GKUpPyieSkbskWv2gAAAAAwlIYyiJbJRe850Ib/XGQ/2c9MjnQgFmn3PBV91vLbftvkqjTeLP6KcBj4r8n82001R0/8edAAAAAAgME1nEG0Urm4Iw0/IE5Vl/XILdIlqX2yroZ6wuB7TObNVehq85rU9tGRV6kctNgOpu8G63bgf3NexkzAHzz/eflCf+YLAAAAACjM0AbRXkDsiLMdPvLJVm/rdsbho6FCFVl0HNlpxr/ltVVeDR4/1SMVLMen7fe6Xbmolqkd3oruc5d1v+2hd5kvAAAAAKAYxxzdo5VFX8HUYoOBQpC/0E/kL+Do4bhHP5G/APjs8mCIr0QDAAAAAHCwCKKB3dxfcX910mn2BrfHj762rJj9fez1sAPAYnSk+bqZdsbH7+knDfj57+j1cdD7vgg7bLRS4fuz31Lyy6OmzPrrdPmgH+Ko9snQbccBRL1yhFHHDA7qGOqY/SGIBgCMBhOcBB022kyHjnS42CN3207JknkLAEcOdUz/DGEdQxANABh++tfzyt7V71JlyB8nOF6TVUd3nFlAh5SZtWUlw7YFgJFFHdNHw1nHEEQDAIac/ax/Za4hO/5JgE6RpzisyfzPD/oWNQDA8KKOQTeCaCC3sB2N257Nat/mpqANidXeRqe0dh7x77sprT2T3YZFJXea1ny65hEbX6eUNi5226gwZWtXdeTl2oe+jPkjTdc803/93n/7rYR8pFJ3e854XoytY9Z8nTffPWrJqnnWv3ty80lNIg8r1L+sm5Oc1EcKJuzDXdur2u3G/JR0bFnTnb3RjqynfdtfdB/tciUjpb1a+H3vu/F9nniLYdI6xOftLr91i525ZTGaf+L7OX3bUc6kCbch9Qq65Nqfvox5JU3XPKljXNQx1DE+J0YPShgMFGLo8le7HixztbljBq47dTMsNc1VHVWYJgxvOP5UtJ1mtXucIFWdxrYZ0ZUyX3tekenvspyx5VhfThgnSHU1peHgL3PvrG0W20Zpsu/DHacxlzSOnaLb2p62qpgDu83THm/PeWZaxz3y+7K9xFmOjfg8e/lOVNp2ymrX/J8w7933eex4scqQ5PFy5ovtRni8W9t+92NYp1h5krpcXgq2Y9p4/naxlyeeInljr2XsvZzxpzE0qFdU6n1/HzR/mYthbb+EsiVJ9v1JHeOmrnn28p2otO2UFXVMdxqmOsbmT8997f612B8CRRu6/JXhZCcYHj/Qg4LRLsDsQsaajlUQ2IVnOM9o4ZB64mUVxuH4VsGx1/rYhbldsMUKqkHlL2/vUrZFqjz7MFaRJe6r6D5PrLitfJY87b0rw7Q8liQcNyXvRiqmWH4M5mkPj1a0kUovGD99WyVJXsaMUvJ56jayt2mW48UenrB89nzS8kT+E5xw/OS8ZW3fLOtg77/Ivkiejr1O4QlnyjQKKGf87w8Na53DfRI9doLh1CuHzl/eYqRsl1R59md6uZm8n1OOVeqYLsnLmFFKnk/dRtQx1r5Ino69TgdRx9j8abmv3b8W+0OgaEOXv6yDLywc0gp3u1BOL8TDAz5FrnkqdgHhFxyJlaAnWJaEcXuqIAaIvx16l1aQ55S4D9PzR9p8kyqKMC/F8oG1H4M8ZueN+Pj7Yq+LPd30fLpnpbfL+uy1L8Jtkjf/pq2HlvzZbvNK/MzeB10VeMY8YS9bhhOcSBmTMn6i1HFTjgtr3aLlmrVe/nTsaefeT7vzl2FoJJYPacdCeh5J3edJcs1TsfNtQl1BvdKrlGMpr8T9mbE8seZLHWM+22NfJJbtmaSth5b82W7zSvzM3gfUMWb8vPspO395NNpEA/sxV5ayeRl1Rsrj5qVSPqkO7T2ZNh9JPRQ+2pIN81KWpyXS2mZyWlThE6XG9zvAWKuNRdqGBI9muLklW/r/+LTMzrlDlDWZPxGOu2t7HSTYZR9GRPOHyiFS9veBv18SdWTrc/NSlmTK7Cc3nQg7PdnYMvstkjei4/f6GA6vvdGYzPvtw9KkHhuW3fL1eFltpT6z27nF5y8lmX7NP26XpOVur7a0/ONnblamI/tQpHLW39prsnqv+9ipnoxvEbWvg3Z28elZeSKXaixvZWHaC1p5KIvOVrD3ZKkS5q1I/vh8y2vXRjmTHfUKUlHHBKhjzOsQdYw7RDmYMocgGjhMkY4mDvr5eBuy5XbuUJLaJ+vdJ0yKf6LUa2V4JPSyD7sq/5KUXzQvC7D2hX+KVJHFSK+hIa9C2qVzkUC0Q5axWp4q8GCEwUTyiQUSRDp8yXDC2qvghJ1y5sBQr4wW6phDRx3TgyNQxxBEA4dFFzD2r8nL6/r+EHHaSUVA76rNHW+6XWlVasGviaoiNMPXl80gy1KF3lQT9boPu64E2L/+Z1WX9WBfxpLdM2jwzMcdaXT96rwm82/v3oNq+7Jd+VWlsZ02rcNTenk2OIlb+7SVvD6mQucqmNaWFfuKgP+4lpST4SzqukdaOw8GadG6+kI503fUK6OFOmYgUMfkdTTqGIJo4JB07q0GBYxbOOz2UHv7tqNLrWghcL/V/cu0Gj8o8INfjLOpXPQLJrsS868uwJZrH0bEt6d9y9Vut6jZVxPy7hP9K62/b61fa3e9tc++tUxXgvYJcgF2y9f2bXh7sW/j0o/H6Hq0Slihu79I+5/b34vPX43R+jTYuzI9qf9XZNqvmG+uSiu2/dt3/COxKrMvRx6AksK6na5relaeKJpVZrjBUPxxLRmVyuHNkMHtnRlRzvQH9cpooY7ZJ+oY6pg+ljkE0cAACAsHVRAntnWyClZVNF0JfulMGT9ScE+Ft7CYX0rd22v85//Zt4pFnkFoF7Dx9lVHgP+swsTUfYva3vvQFv11vn05vEWv+tr0rpWN3SYqOo3uZbOflxj5ddw+ecjSrkyzToQ6N84VdGtWWr7uSPPtPG2n1Mnbh9Yv3F37zr4FsiqND/0KPdombcrK//Y6VpsLakk9adtfH0dBu9DlCxlPBO35q+n9PJy/nSf6KQyG8m5zxWoPuVY7Fx4TVpkS5DvKmQNHvTLgqGOiqGNc1DGWQa9jVJQeoQclDAYKMXT5K7EHzJReBFN6WtQSe42M9CSYktJ6L7TTXMrzPHedfrTnQrvXxaQULPOA85e3dynbuCuZ7ZdrH9r5Iy1lyDfKrvsrkmf2mufePVjulTei00g7NtLXJdM236Pn1ECkp9jkFB7HoV3XMWHe9rp0p1gPrIlliC1nvkjp3TRch9g+TRw/7zZPWEb/8123eXRb9Kuc8b8/NKhX3HR06hVbhmPPTdQxYaKOiSbqmDAdTB1j86elcSUaOCy6HVFX+xCvDZIqCDyR2390W49Ypwm6fdQnF8LblWxuO6WEdkUJt0y5t70ktrHy2icturcZoUvufejT48T2pbtf7LY96dL2l6ogYrf7ebfXqQrZvLfpZdj71jk9r2BdDFVRW22b9tvRis7X8XyasH2ymFzUNVtiWyhv+zqy+kb3NRh3eya01XK3Z8JtaKU3VtV8upfP3S4Z92HI20fRZdbHXcKxW5iEssQc60Feidz6p5cxNr5/1cjd5gn7ym27Gd0WlDMHgHpltFDHKNQxPuoYywDUMcd0JG1eu/RlcC02GCgE+asf2rLi30qkC5XMbaZGD/kLOHo47vuBesVH/gLgs8sDrkQDQyH6CAi77VHnxpWgXUv3MwIBAEhCvQIAveJKNA4U+at3ugOP3Z+fqG9Pynubz2ghfwFHD8d976hX9kb+AuCzywOuRANDwm0jk9CmxpXQNgQAgN1QrwBAb7gSjQNF/kI/kb+Ao4fjHv1E/gLgs8sDrkQDAAAAAJDRsd9vb0d+WvvaiRPmFQAAAAAA8Kn4mSvRAAAAAABkRZtoHCjyF/qJ/AUcPRz36CfyFwCfXR5wJRoAAAAAgIwIogEAAAAAyIggGgAAAACAjAiiAQAAAADIiCAaAAAAAICMCKIBAAAAAMiIIBoAAAAAgIwIogEAAAAAyOgAgui2rBw75j6cOjVdbptxC3B/RWZvdMybgjxqymzScsdTkesxMDrSfH1F7UUAGBydX63I9X81b/J68lCa/3RblW4AAHSjjsFeBuNK9KUpFYTuP1BrX1aBbGXJvMO+uT8ejMn8TfMeAAbB/RUZm16Szp/M+1w60vz+aZm/98S8BwDAQh2DDAbodu4lmXq92fuvNirDT10yr1EAVQi8PS9r5h0AAAAA4KCD6LmG7DiOOHZq182Hys15uX7fvB5Uy+vR5bfTxYoZCQAAAAAwig7/SvTkYiSQXroTu6n7/kp32+PYrd/x27jXamPueJG20Rmm0w/uskXmOSvNR+bDgG537H+ulsleVn11PnjvfTc+zRX/h4fYOia3DbfnlTJu123cSzLljkfb6P0z21/t14e/bcqF75yWMb1tvzkl37u8Jg//aEYz3H2txt3S7WvefVVOj+n9MCanv3NBmv+WcKvQk025/kN/vG/I1N99IK3OVjBP2udg2Nnl/VJF53NdLj6T1ru63B+Tt/5v9Lh4pspFXX6NvduSZ3bZdnPeO/ZGsi8LHC3UK0BRqGOQmROjByUM3od1R4XI3nTnGs6OGRqVPM5OsxosT3eqOo1tb7z15aTPxak2vSllnU6q7YZT9cdfXjcD92KtU0Lyl82z4zTmdhmvXU/8zE7VueR1jM5n92UK1s1e30iqqynsjz+to8vs65cqTqVUcqb//qqz2rrrrL5fdcp620zWnY2nZlTFzdtna05tMhz3VrPuVE/q7VhxrmyaEbXtVWdBDy9NOws/XXXutladq29W1Hs1r5fU8NTjb3SQv0bf44d3nbsfLbj7eeEj9bq14ezoY+aru875ks7/5527X3njOl+tO4s677+06KzrYU93nA11XCyeVcPOLrrH092Hj71xMbQ47qlX+on8dbRQx2A3dnnQVSoUX1hkCaLtINIP1KzvWYGrHRBHAkQr0EwNHLNMJ0lqUGmlWHBtB/b1thkYWU87eI8F0fFAPRJEh9+L/zgQrIe9vIk/StjztgNrO0hO2if7502Pkx29DSpqP1vnNc7TfznvlNTwmZ99aYaE+Sg+rrN5xZlQwyc+3DADnjp33yupcdW+3TKDXE/VNNQJjxqXkx2MDFMmhmWrxz+GJtzjReX9ixNqvIoazz56zDF4BI6Ho4Ljnnqln8hfRxB1DFLY5cEAdSwWV5FFtax6ee22xqXyGfMqq6Kmk8OjplzxOzlbXpfFSfNaSlL7sCEqwFXWZP7nSbd4VKXx5i5tq5cvSG3ce1l6edZMS5lryLU31KGtjZclWLubW7JlXpbeWPW2g7MaTEOkLOU58xIH6LzU36nIcfNOO372nNTPitz+6ZpsmmGeaVl4IzquvDQlM+rf5h+fee+ftGTtHzsi75yT6te9QZ7jUvnBBVkw74BRdvxsXRrvlGTz0nn54H/W5dzlTaksfyDnJyNHDzCiqFeAfqKOgW2Ag+g487zpfT/CqqjppOvcWw16ta6fjQXE49My6wetl1oJbYzPSDkIcLtVT5bNq5gXyypE92ULjDs3Zt32HTzC6hDMnZHTXWVuWU6/rP59tiFbkUZmp2TsBfPS8hf6z85jcVvnfLEpq+pfdeJ09KRIe6Esp14yr4GRdlymLzXkfKktS2+vyOZkXa7FggpgZFGvAH1GHYPQgATRW7KVFMhFOsqakp7D3qKmo6X1zj0UPXNHOxUbq/EAq0PzV/aPHnHP5Fnk2YRj8vxz5mXcH56osRU1fnrnLmMy9k3zEhh1fzklM2+ecl+WXq7IKc5ucFRQrwD9Rx0DYzCC6PutMLBdnhY3HNU93NlXi/3g1X4kVhZFTWcEtC/bV52r0tjWPwDsSIPbuQ/eb7eST07+U/8Zk+f+0n2X3Z/rxgJpnsrThA5XgVH07Nd1ufCPD6V8siydn1yQ+q/NranAqKNeAfqOOga+ww+iY0Guf/tz5Jbodu9XeouaTh52W+WuR3Y9asmqH8j6PxgciLa0/Hba7vO67XbROHD3wrbqoS15oB9XdnZCTqVdIUhzckJm1b+1zQfeFQTbHzalfce8BkbZH9tS/5F3i91qe1Xqk5uy8qO6tGOP+AFGEvUK0F/UMbAcbBDtPzPNTifmgyBXB3cLQSdcoY2gIU9bVvbRlrmo6ewp0u55KnyOs76d+m1/fffoQKyfrM7GOjfO0Sb6MHSuytUb0dOdJ7+8JkvqpKT65quS0vI93XPTUn2vJPKTa7L2OzPM9Uza/+uKXDfvgFHyNHJ7qsrr75+Tlc8mZPEn52XihQk5/5NFmfhsRc693+4OAtR3n5qXwEigXgEKRR2D3QxQx2J1Wf+kFtw6ZF/NXavpB5zroDvannntC6uyGC93jT97o5N/OoUoSe2TdbVGHu9h7TqFt1NXm9cO+EpwRaaXzUu19vrB8HqZou2iN2TrkXmp1qH8onkZjL+S0BEa0vlt0PWD+s0gn8rordorMnu5Kbd/dVual2flzMwHIt9tyBW/l/Vcjsv0D1WeKq3JfOUV+d4/rUnrV2ty7fuvSvWjrV1uyQMGW/uyV1aFP0YqL5RkWv27/s/X1PGzKR119qJvsdM9pZbeuSJLf+M1Ujv+N0tyRfekevmcdcvdc1LSPQ1/elWu32hJ67fck4phQr0CFIk6Br0aiCC62twRx1mM3to8XpPVbf9xUD4VaDuOrPvBoN27tR4/1s7ZDY7zTqcw3qO1dprROfvzXu2pQtufykVrnQ132wfbZ01W74UtqioX4+2l7SAb+/LyB9J6WJfyb+qyMP2qnP/kmcz8dF0e/qKW/2qBb7wqjc/X5eprx2X9x7PyyvQFafxpRhqbDXlLf/71kiragRFwcl6u/LQmz995S16dnpHr/8/cYlc6J9ffnbZ6SlVBwLvX5VzJvuVOBwa35Py3OrKiAo5XftxKbkcKDBvqFaAY1DHI4Jije9my6F9jtNhgoBDkL30VYUzmpSE71p0XffWsJRf+4hVp/HhDdt6bMANHE/kLOHo47qlX+on8BcBnlwcDdDs3gN5tyso3T8sr7972nu9peXJnTZ1alWRh0nskAwAAe6NeAYA0BNHASDglM3/3nLR+siCz3/9Amr9suW3Xrr87K2e+c03kux/Iwt/yMEMAQFbUKwCQhiAaGAnHZeK92/Jg7S0pf3Fdzs+8Iq9Mz8rKvedlYe3B/trEAQCOIOoVAEhDm2gcKPIX+on8BRw9HPfoJ/IXAJ9dHnAlGgAAAACAjI79fns78tPa106cMK8AAAAAAIBPxc9ciQYAAAAAICvaRONAkb/QT+Qv4OjhuEc/kb8A+OzygCvRAAAAAABkRBANAAAAAEBGBNEAAAAAAGREEA0AAAAAQEYE0QAAAAAAZEQQDQAAAABARgTRAAAAAABkRBANAAAAAEBGBx5Ety8fcx9UnZ5WpG3G3b+2rLzelI55V5S910GlPswXB+2JPLxxQV59cczdp2MvvioXbjxUQ/voTx1p/eR7MvVNLx99o/I9Wfn0oTwzH0flW74n/9aUC985LWM6f46dllffbcrDlJHzjAtEJOThD37V59LwyUNpvvuqnB7T8xyT0995S679OiXD5lq+PMfYIZQXGEKHnU825YPKMZm9UUSeV2NTr+BAcNxw3AwgJ0YPShhcmPVlb/p7pXrbfKFX7bo3rbmGs2MGFSXrOohUnca2+RJc/rYZfE/Vfq6oZS071eWGc6t1y2m8N+OU1LJXltfVp33w1bpTn9Tbp+TMvGfmuVx1yonzzLd8T9XxUFGflefqTuP2XedWc9GZKal5Tdad9a/MSEaecQfN8OSvUfWl0/huyZHStLPw01XnbmvVufqmzqclp9b80oxTsK2GU9P5UyrhPP9+Wh0LSfPMs3x5jrFDKC8QoF7Jyp+/ONVm0pkR9UoS6pXDxnEzjMfNqLLLg65Sod+FxcEEoCoY8adzqEG0SuoAQsjfLoNOF1wTajmrv4ieWH/5i6pa/gnnyqYZUKCND3Uh3H0y/2WzpgrjknPu/zw2Q3Iu31N1PLyktrs6FiJjq+CjqqYx8f6GGaDkGXcADUv+GlU7TS//Ld6zTxvUCcbFCTV8wVn9DzOoMI+dWz9QQbEKoOvt2Dzdk5pK5FjIs3x5jrHDKC8Qol7J4rGqY7wTe72tkoIB6pVk1CuHi+NmOI+bUWWXB12lQr8LizAATQqSVYYx83dTzwGoNR2V+foXRKcE+uqACtahD/MfZv52GWxPnbvv6RPz887drp8Q7zrn1fKX3rvb9evi/mw4df3r4tmrzgMzJLThXDEFqZeX8i3f09aiWwGc/5eukZ2776jplhaD6eQZdxANR/4aVV86H39L5+GPoxW+9u8fO9Nqv8z8rOuT/dlZdWp6n//oVvfx+PiWc059VgrqkTzLl+cYO4zyAjbqlT3s3HWuzJXVvEtO7bv6xD4pGMi3fNQrOBgcN8N63IwquzwYsI7FKrLorIsKgD2XWrH20R1pvu61Y7NTpI3C/RU1bEqWzFu5Oe+1I7DbKD9qymxsGseOzUrzkfl8vyYXpDFnXt/cki3z0pV13vZ4l9VWiH8vrc11bDx327jbxHof0ZYVa/zkcY6ah7L5S7UNXjsj5eNmkO/4aTmj9m3nl5tqrAI9eigberO/fEZOeUMsEzL1bfXv5oY8cBtH51u+h5/dVnmlKme+2TWynJ6oqpFvy+ZvvSF5xgUinuh8qf5PnpayNyR08rSoIFVu/6bgNmy/eyhN9a9aOaNyaMxzp2TqNZVl7z3wyuBcy5fnGDuE8gJD6LDyiTpv+odX5MLNkpxbW5fGD8+Y4XH5lo96BQeD44bjZnANYO/cFZleNi9lQ7aC4FIHe2Myf9O8tazVxrxAMwsdUJ6YlzXzNrQm8yeOycp983Y/7l8Pl3OuHJ6w9TrvS1Pd39M/DsTXOWH6etvMfrhh3sW4Abf1g4ORa3uOomePZecz9f+vrX0XKEn5r9S/z7Zk58A7dtiSzh/Uv1zL90we72yq/6ek/HX9Pqp0Qofsm7K1rUfOMy4Q84cd90Sh+s2S9z5C5UsV0Er7YfRHxYNwp6NORJQ8y5fnGBvY8gID5RDzyfEXr8jdnXW5+lr3nAPUKxhEHDccNwNsIB9xVT5ZNa/WgiC6c+OKCfaq0th2b0NXKeGq9eRidPhcQ3b0uJ/UVHbuSPNDEzL6w3VqB2PL0p08waMX/NpXcd1UCcPS6mvTar7a/uZdbe5442831BYwIlfq27ISzNfaRmr6aze7w3Z3ed42Abe1PDtNM3UVuBfyg8Iw+kPHO5H+L+67bu7wHXnyR/ddMcbLqqhUPleBsjvAtiUP7H2Ra/meSOd3+v9f6D/d/twbvvNHfYk7z7hAjMqXLf3f5JMoNezP1b/PnqlTgwKVym55uP5FQmj+bEs2PjWvtTzLl+cYO4zyAsPn0PJJSaoXz8t00m9HtlzLR72CA8Jxo/9047gZCEPznOjSG6teUOisSm3cDJSylOfMy0xKUvvEBJduUG2oACYITIu0vC6rb/hz2ce8VZB7zZ/OeE0uJF2pv98KrygvXwi30eSirAfjW6yr5fW3w+UpvXEtuBU93w8K2J8JmX5P7YWbV+X6/Wih+OzX1+XaHfMGQOjrFZn5lkjno6vSdE84Qls3r8kH5jUAAECRhiaIjuvcmBX9PNCk27uz63htrBNvse5dcNX4YsUMSZJj3i+Ww6A7RWcrvGW7fjY638rZ8Gq3zx5/qWJdRbe3aeJVUfTHcZn+H2tSn2yr/XFaZi835favbkvz8qycfrklp76rW2wCiCrLwkcNqUlT5iuvyPf+aU1av1qTa99/RabePS61XD+yAgAAZDOQQfTWFym3H1udio3Vegx7Ix1v7TcIt26btm6zdtshJ3XOVei8D0C8U7Sj4oWS17blP9133dzhZSm94L5L5P3I4+9rk9I6g/P9ZUUWf/lAGu+ckocfzcur02/J1a0p+XhnXepnn1MjHJfj+pbTXMv3nJTc9jRP9Z9uf/KGl1/Q088zLhCj8qX7U4/JJ1Fq2J/Uv7OlXX4QjJbxftqzo8Ov16SxeVeufOuptP5hVl6pLcnt5xak9bAhb+n2aGqGbrcseZYvzzFWQHmBI6CAfNJTvZJVruWjXsEB4bjRf7px3AyEAQyi29K6ZF5KXaYnvVfty3bQ6QevO2Ev2Jm0ZcW+8uu3BbbbGfdqvCarVvtmHUhH2xT3cd77VG+bHwK60qLsdi19ZB1/XsZeUv9/m3QlviNbujfEl8bk+XiHiUV47pTU3r8lD3b09v9S1n+m2+Q8k63PVc556YyU9Rl+ruU7Ls+PTaj/D61O+kKdbd3V0oSM/Vc9cp5xgZgXxtx2/Wu/Szo1UflSt08+qfKlN6BYpWk5/7N1+VKXWzsP5Nb7NTn1nOlL4LUJ7yQnz/LlOcYOs7zA8Bj0fEK9gkHEccNxM8AGLIjWVyKs3qKXp00QZwXWbvBpt4vOwWo37N5ybbdNLkKs/fFSZUUtudHneZfKYff78bbM7TvBFg3Y429sdR/6R9spmfiW2jvBI6Uszx7Ixk21/b414XUEliJsw2+lXfd5R27/8LSM/ffr3Vf/n61L63+raVanVJGp5Vu+Uy/NqPmuycbDrpHlwaYKzkszMqGv2Cl5xgUintP5Uv33Hyll++KB26nXzH87Jem/m1v9Rlgp7FciyaZcq3xDjr3b6u6w7IuWrN1R5e3LZ7x55lq+PMfY/ssLHAX7zyf565U88i0f9QoOBscNx80AU5kpQg9KGFwYFWQG89g9VZ3GtvmSs+7Ug+F19c6z0/Qefh4fHhl/ruHsmKFOux6OvxxMxWnMmWGR4enCdbCX0WcvqzW9vPPebjjVpOFK8vzt+VrD7fmqFD4ofu/xw3GL40970D1V20EFrE71F1+aIZ4v/3lGLf+Ec2XTDCjQxocTatqV2LSfquF6njPOx/9uBim5lu+p2tcvqe2ujoXI2P/+sTOjpjHx/oYZoOQZdwANS/4aVV6ZPOEs3ntqhmhPnbvvlNTwBWf1P8ygwjx2bv1A7fPSOedWZNqPndUfqHmWzjt3vzKDlDzLl+cYO4zyAiHqlRzUMuhtlVS/U68ko145XBw3w3ncjCq7POgqFfpdWGQNoutt8wVj7+/ZAWUsONVJZcKdeICblNzxdrd7EK2Ygy06Ts555w6ilch8w1SdC39siBQAKeN7yf5Rojj+9AffU7WdK2pZS87Mew3nVuuW03hvximpZa+o/WGfghfmK5VHJtX2OVl16s1bzt3WqnP176fVPEtOrRktnPMuny7kK+qz0rcXncbtu86t5qIzU1LzmlT72QoytDzjDprhyV+j6kun8V0dkFachZ+uenn4TS+fdufhgmw1nJrKn6W/XXCurt117t5uOPW5srsM9Xb8SMizfHmOsUMoLxCgXsnB1PvJP5JTryShXjlsHDfDeNyMKrs86CoV+l1Y7BkM7xLExr/rZmYr2Ixm7njQqjJc4nAvEA2vaqcExpY9g2glsqzBOuWYdy9BtGYOdD+5P0ZYw7oLgITgPja/IvnzGA6PnY2fnXNm/lqfdKtC7K9nnPPNB2poHz3ecD7+wYxzSheSqkA+9e1zztV7aXPMt3yPf/Oxc+7bp9yCXUqnnJl3Gs6DlJHzjDtIhit/jaj//NK5tbzgVE56+6I8ueBcaaWV6gXZuuXUVTBcdvd/2am8WXdWH6acWuVavjzH2CGUF3Dp7a3TcDjkfLJrMKBRr8QNV/4aVRw3w3bcjCq7PDhmBgR0r3VabDCG2f0VOVbx2kXrTsQWTWdth4H8hX4ifwFHD8c9+on8BcBnlwdD+5xoxEQenzUrzaA3v440Pwy6NJNyLx2yAQAAAABcXIkeGbpn8z2ePb28Ls7FinlzOMhf6CfyF3D0cNyjn8hfAHx2ecCV6JHhPR5mp5n81Gn3WdCHHEADAAAAwLDjSjQOFPkL/UT+Ao4ejnv0E/kLgM8uD7gSDQAAAABARsd+v70d+WntaydOmFcAAAAAAMCn4meuRAMAAAAAkFVXm+jtR96zkU6M8ywkFI/8hX4ifwFHD8c9+on8BcBnlwdciQYAAAAAIBOR/w+Hjj3sYHjhqAAAAABJRU5ErkJggg==)
# <br>
# - Each image has its corresponding label and bounding box in the labels folder. 
# - For example: **0.png** has its label and coordinates of the bounding box in the **0.txt** file.
# - The shape of an image in the dataset is hence **(300,300,3)**.

# %% [markdown]
# ## **Objective**

# %% [markdown]
# **Object Detection** is **a combination of classification and regression.** 

# %%
# Mounting the drive to load the data from drive
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## **Unzipping the Data**

# %%
# Unzipping the zip file which contains the data
# !unzip -q "/content/drive/MyDrive/Advanced CNN Recordings/Datasets/MNIST_Object_Detection.zip"

# %% [markdown]
# ## **Importing the Libraries**

# %%
import os                                 # Importing os module to get the path
from os import listdir                    # Importing listdir module to get the list of directories or files
import numpy as np                        # Imporing numpy for numerical operations
import pandas as pd                       # importing pandas to read csv or txt files
import matplotlib.pyplot as plt           # Importing Matplolib for ploting graphs and visualizing images
import cv2                                # importing opencv to work on images
import seaborn as sns                     # importing seaborn for data visualization 
import tensorflow as tf                   # Importing tensorflow for tensor operations or model building
import keras                              # Importing keras for model building
from tensorflow.keras.models import Sequential, Model                 # sequential api for sequential model 
from tensorflow.keras.layers import Dense, Dropout, Flatten           # Importing different layers like dense and flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input, Activation # importing CNN layers and activations
from tensorflow.keras import backend                                  # Importing backend module to clear backend 
from tensorflow.keras.utils import to_categorical                     # Importing to_categorical to perform one-hot encoding 
from tensorflow.keras.optimizers import RMSprop,Adam,SGD #Importing optimiers for optimizing the model
from tensorflow.keras.callbacks import EarlyStopping  # regularization method to prevent the overfitting
from tensorflow.keras import losses, optimizers       # importing losses and optimizers from keras
from google.colab.patches import cv2_imshow           # importing cv2_imshow to display images

from sklearn.metrics import mean_absolute_error,mean_squared_error     # importing the error metrics for regression
from sklearn.metrics import accuracy_score, confusion_matrix           # importing the error metrics for classification

# Importing the transfer learning model- VGG16
from tensorflow.keras.applications import VGG16

# %% [markdown]
# ## **Function to store training data in a list**

# %%
# Read the data 

def create_training_data():
  training_data = []                                                            # Creating a list to store the training data
  IMG_SIZE=300                                                                  # initializing image size to 300
  # get the path/directory
  folder_dir = "/content/mnist_detection/train/images"                          # path for training images directory
  dirpath = "/content/mnist_detection/train/labels"                             # path for training labels directory
  
  list_dir=[int(file.split(".")[0]) for file in os.listdir(folder_dir)]         # separating the image name from extension and storing it in a list like 0.png is stored as 0 and 99.png is stored as 99.
  list_dir.sort()                                                               # sorting the image names in ascending order starting from 0

  label_dir=[int(file.split(".")[0]) for file in os.listdir(dirpath)]           # separating the label name from extension and storing it in a list like 0.txt is stored as 0 and 99.txt is stored as 99.
  label_dir.sort()                                                              # sorting the label names in ascending order starting from 0

  for images,filename,i in zip(list_dir,label_dir,range(10000)): 
            data = pd.read_csv((os.path.join(dirpath+'/' + str(filename)+".txt")), sep=',',  header=0)      # Reading each txt file with header as 0th row and each value is separated by comma.
           
            # Storing the four bounding boxes
            xmin=data['xmin'].values[0]
            xmax=data['xmax'].values[0]
            ymin=data['ymin'].values[0]
            ymax=data['ymax'].values[0]

            img_array = cv2.imread(os.path.join(folder_dir+'/' + str(images)+".png"))                       # Reading the image
            # Resizing each image to image_size=300 as we require each image to be of the same size before training the model without missing any image of a different size
            new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))                                          
            training_data.append([new_array,data['label'].values[0],(xmin,ymin,xmax,ymax)])                 # Appending the image array, 
  return training_data

# calling the create_training_data function
training_data=create_training_data()            
 

# %% [markdown]
# ## **Function to create testing data**

# %%
# Creating test data similar to train data
def create_testing_data():
  testing_data = []   
  IMG_SIZE=300
  # get the path/directory
  folder_dir = "/content/mnist_detection/test/images"
  dirpath = "/content/mnist_detection/test/labels"

  list_dir=[int(file.split(".")[0]) for file in os.listdir(folder_dir)]
  list_dir.sort()


  label_dir=[int(file.split(".")[0]) for file in os.listdir(dirpath)]
  label_dir.sort()

  for images,filename,i in zip(list_dir,label_dir,range(10000)): 
            data = pd.read_csv((os.path.join(dirpath+'/' + str(filename)+".txt")), sep=',',  header=0)
          
            xmin=data['xmin'].values[0]
            xmax=data['xmax'].values[0]
            ymin=data['ymin'].values[0]
            ymax=data['ymax'].values[0]

            img_array = cv2.imread(os.path.join(folder_dir+'/' + str(images)+".png"))                    
            new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))                                         
            testing_data.append([new_array,data['label'].values[0],(xmin,ymin,xmax,ymax)])  
  return testing_data

testing_data=create_testing_data() 

# %% [markdown]
# ## **Splitting the images, labels and bounding boxes**

# %%
X_train = []                                                                   
y_train = []
X_test = []                                                                   
y_test = []
y_train_boxes=[]
y_test_boxes=[]

for features,labels,boxes in training_data:                                     # Iterating over the training data which is generated from the create_training_data() function 
    X_train.append(features)                                                    # Appending images into X_train
    y_train.append(labels)                                                      # Appending labels to y_train
    y_train_boxes.append(boxes)                                                 # Appending bounding boxes to y_train_boxes
   
for features,labels,boxes in testing_data:                                      # Iterating over the testing data which is generated from the create_testing_data() function 
    X_test.append(features)                                                     # Appending images into X_test
    y_test.append(labels)                                                       # Appending labels to y_test
    y_test_boxes.append(boxes)                                                  # Appending bounding boxes to y_test_boxes

# %%
# Converting the lists to numpy arrays and reshaping them to (300,300,3)
X_train=np.array(X_train).reshape(np.array(X_train).shape[0],300,300,3)
X_test=np.array(X_test).reshape(np.array(X_test).shape[0],300,300,3)

# %%
X_train.shape,X_test.shape

# %% [markdown]
# ## **Preprocessing the Data**

# %%
# scaling bounding box relative to image size
y_train_boxes_scaled=[]    # list to stored the scaled bounding boxes
y_test_boxes_scaled=[]     # list to stored the scaled bounding boxes
IMG_SIZE=300

# Scaling the train_boxes
for boxes in y_train_boxes:
  value = tuple(i/IMG_SIZE for i in boxes)
  y_train_boxes_scaled.append(value)

# Scaling the test_boxes
for boxes in y_test_boxes:
  value = tuple(i/IMG_SIZE for i in boxes)
  y_test_boxes_scaled.append(value)

# %% [markdown]
# ## <b> Visualizing the Images

# %% [markdown]
# <b> Plotting the images with their corresponding labels

# %%
# f = plt.figure(figsize=(100,100))
f, axarr = plt.subplots(2,2,figsize=(10,10))
axarr[0,0].imshow(X_train[100])
axarr[0,0].set_title("Image corresponds to: " + str(y_train[100]),{'fontsize':15})
axarr[0,1].imshow(X_train[200])
axarr[0,1].set_title("Image corresponds to: " + str(y_train[200]),{'fontsize':15})
axarr[1,0].imshow(X_train[300])
axarr[1,0].set_title("Image corresponds to: " + str(y_train[300]),{'fontsize':15})
axarr[1,1].imshow(X_train[400])
axarr[1,1].set_title("Image corresponds to: " + str(y_train[400]),{'fontsize':15})

# %% [markdown]
# ### **Observations**
# - We observe that each image consists of one number.
# - The number may be present at different regions in the image, such as the bottom, top, left corner etc.
# - We also observe the corresponding labels at the top of each image.

# %% [markdown]
# ## <b> Visualizing the Bounding Boxes

# %%
def plot_boxes(index):
  (startX, endX,startY, endY) = y_train_boxes_scaled[index]                     # Getting the coordinates of the boxes
  images = X_train[index]                                                       # Getting the image
  (h, w,_) = X_train[0].shape                                                   # Getting the height and width
 
  # Getting the coordinates to original height and width
  startX = int(startX*w)
  startY = int(startY*h)
  endX = int(endX*w)
  endY = int(endY*h)
 
  label = y_train[index]                                                        # Getting the label
  imgHeight, imgWidth, _ = images.shape                                         # Getting the height and width of the image
  thick = int((imgHeight + imgWidth) // 900)                                    # Giving a random thickness value for the rectangle
  color = (0,255,0)                                                             # Giving a color to the rectangle box
  
  # plotting the rectangle on the image with title
  cv2.rectangle(images,(startX, startY), (endX, endY), (0,255,0), thick)            # Plotting the rectangle box on the image
  cv2.putText(images, str(label), (startX, startY - 12), 0, 1* imgHeight, color)    
  plt.title("Image corresponds to: " + str(label),{'fontsize':15})
  plt.imshow(images)



# %%
plot_boxes(10)

# %%
plot_boxes(55)

# %%
plot_boxes(565)

# %%
plot_boxes(88)

# %% [markdown]
# ## <b>Note: Images with Inaccurate Bounding Boxes

# %%
plot_boxes(250)

# %%
plot_boxes(350)

# %%
plot_boxes(450)

# %%
plot_boxes(650)

# %% [markdown]
# ## **Converting the list into a DataFrame**

# %%
# Converting the list into DataFrame to define the label column so that this can be used along with its axis during predictions and the dataframe operations can be easily implemented
y_train = pd.DataFrame(y_train, columns=["Label"],dtype=object) 
y_test = pd.DataFrame(y_test, columns=["Label"],dtype=object)

# %%
# Storing the value counts of target variable
count=y_train.Label.value_counts()
print(count)
print('*'*10)
count=y_train.Label.value_counts(normalize=True)
print(count)

# %% [markdown]
# ## <b> Converting the target features into arrays

# %%
y_train=y_train.astype('float')
y_test=y_test.astype('float')

# Converting the pixel values into Numpy array as arrays are computationally faster as compared to lists
y_train_boxes_scaled= np.array(y_train_boxes_scaled,dtype='float32') 
y_test_boxes_scaled= np.array(y_test_boxes_scaled,dtype='float32') 


# %% [markdown]
# Over here, we delete the unrequired variables to save memory, as this would also help in training the model without resulting in an error due to memory exhaust.

# %%
# Deleting the y_train_boxes and y_test_boxes variables to use the memory/RAM efficiently.
del y_train_boxes
del y_test_boxes

# %% [markdown]
# ## <b> Initializing Random Seeds

# %% [markdown]
# In this step, we fix the seed for random number generators so that we can ensure we receive the same output every time.

# %%
from tensorflow.keras import backend
backend.clear_session()
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)

# %% [markdown]
# ## <b> Building the Model</b>
# Here there will be 2 sections in the model-building process:
# - The first section is for classification
# - The second one is for regression / object detection
# 
# Objection detection = a classification + regression problem.<br>
# 
# So we need a classification layer which would classify the images into 10 categories, and we also need a regression layer which would plot the bounding boxes around the numbers in the images.<br>
# 
# 
# 

# %%
def get_model():

    # Using the VGG16 model to implement transfer learning
    # We are defining include_top=false to remove the dense layers from the VGG16 model
    vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(300, 300, 3)))
    # Freezing all VGG layers so they will *not* be updated during the training process
    vgg.trainable = False
    # flatten the max-pooling output of VGG16 so that we can add the required dense layers to the model.
    flatten = vgg.output
    x = Flatten()(flatten)

    # Adding a classification layer to classify the labels
    classifier_head = Dropout(0.3)(x)
    # Adding a output layer for labels with 10 neurons
    classifier_head = Dense(10, activation='softmax', name='label')(classifier_head)

    # Adding a Regression layer to predict the bounding boxes
    reg_head = Dense(128, activation='relu')(x)
    reg_head = Dense(64, activation='relu')(reg_head)
    reg_head = Dense(32, activation='relu')(reg_head)
    reg_head = Dense(16, activation='relu')(reg_head)
    # Adding the output layer to output 4 regression outputs for bounding boxes
    reg_head = Dense(4, activation='sigmoid', name='bbox')(reg_head)
    
    return Model(inputs=vgg.input, outputs=[classifier_head, reg_head])

# %%
# Calling the get_model() function
model=get_model()

# %%
# Plotting the summary of the model
model.summary()

# %% [markdown]
# ## <b> Initializing the Optimizer and Loss Metrics</b>
# - **SGD with Momentum over Adam**, as it generalizes and gives better performance than Adam, although Adam is computationally faster.
# - As our problem involves both classification and regression, we will use <b>sparse_categorical_entropy</b> as the loss metric for the classification of the labels and we will use <b>mean squared error(mse)</b> as a loss metric for the regression on the bounding box values.

# %%
# Using SGD optimizer with momentum as it generalizes better than other optimizers like Adam
opt = SGD(learning_rate=0.01,momentum=0.94)
model.compile(optimizer=opt, loss=['sparse_categorical_crossentropy','mse'])

# %%
 # plotting the model architecture
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

# %% [markdown]
# - We can observe from the model plot that the model accepts and input of shape (300,300,3) i.e. images with a height and width of (300,300,3).
# - We also observe that **we are using the VGG16 model architecture** before the flatten layer.
# - We observe that after the flatten layer, **we have two sub-branches**: one being used for classification, and the other for regression.
# - The branch on the left-hand side is used for regression to predict the bounding boxes for the images, and the one on the right is used to predict the classification label for each image.

# %%
# Fitting the model with 10 epochs and validation_split as 10%
history=model.fit(X_train, 
          [y_train,y_train_boxes_scaled],  
          epochs=10, 
          batch_size=64,validation_split=0.1)  

# %% [markdown]
# ### <b>Observations

# %% [markdown]
# - Here, **we have used 10% of our Training data as Validation data.** This is separate from the inital 20% of the whole dataset that was kept aside as the Testing dataset. 
# - So our **Train-Validation-Test percentage split** is basically **72-8-20**.
# - Here there are 4 values that we need to consider: bbox_loss and label_loss for the training data, and val_bbox_loss and val_label_loss for validation data.
# - We have defined mse loss for the regression problem of predicting bounding boxes and sparse_categorical_entropy for the prediction of labels, as it is a multi-class classification problem.
# - We observe that **the mse value and the cross_entropy value are decreasing per epoch**, starting from the first epoch to the tenth epoch.
# - This CNN, as we would expect, is more computationally intense to train than the regular CNNs used for image classification, and that needs to be kept in mind when training such models. On a regular CPU, it takes considerable time even to train just the 10 epochs displayed above.
# 
# 

# %%
# Deleting the X_train variable to use the memory/RAM efficiently
# Here we will be deleting the non-required variable to use the memory efficiently and to ignore the memory exhausted error.
del X_train

# %% [markdown]
# ## **Model Evaluation**

# %%
# Obtaining the predictions for labels and bounding boxes on test data
# Here yhat1 hold the prediction of labels in OHE format and yhat2 holds the predictions of bounding boxes
yhat1, yhat2 = model.predict(X_test)

# Obtaining the class from the onehot encoded output
yhat1_classes=np.argmax(yhat1,axis=1)

# calculate Mean_Absolute_Error for regression model/ bounding boxes
mae = mean_absolute_error(y_test_boxes_scaled, yhat2)
print('MAE: %.3f' % mae)

# calculate Mean_Squared_Error for regression model/ bounding boxes
mse = mean_squared_error(y_test_boxes_scaled, yhat2)
print('MSE: %.3f' % mse)

# evaluate accuracy for classification model/ labels
acc = accuracy_score(y_test['Label'], yhat1_classes)
print('Accuracy: %.3f' % acc)

# %% [markdown]
# ### **Observations**
# - We observe that the bounding boxes are predicted with lesser errors - an MAE value of 0.023 and MSE value of 0.001.
# - However, the labels only have an accuracy of about **67%** during prediction on test data. 
# - **This low accuracy is primarily due to low-quality annotations and ground-truth bounding boxes in this object detection dataset.** Even though the MSE is very low (meaning the regression task has ostensibly performed well), that is in relation to the poor ground-truth bounding boxes which don't properly encapsulate the handwritten digit image, and due to that, the classification portion of the CNN is not able to perform well in the region localized by the poor bounding boxes, as from its perspective, the handwritten digits within the bounding boxes are not clearly discernible.

# %%
# Plotting the confusion matrix
cf_matrix = confusion_matrix(y_test['Label'], yhat1_classes)

# Confusion matrix normalized per category true value
cf_matrix_n1 = cf_matrix/np.sum(cf_matrix, axis=1)
plt.figure(figsize=(8,6))
sns.heatmap(cf_matrix_n1, xticklabels=np.unique(y_test['Label']), yticklabels=np.unique(y_test['Label']), annot=True)

# %% [markdown]
# ### Observations
# - We can observe that most of the classes are classified correctly with a 70% accuracy but there are some classes like 2,3 and 5 are classified incorrectly.
# - We can also observe that 6 is sometimes misclassified as 0 and 5 is misclassified as 3.
# 

# %% [markdown]
# ## **Plotting the Output Predictions**

# %%
def plot_predictions(index):
  '''Make bounding box predictions on the input image'''

  label,preds = model.predict(X_test[index].reshape(1,300,300,3))               # reshaping and providing a single input image
  (startX, endX,startY, endY) = preds[0]                                        # Getting the bounding box
  label=np.argmax(label)                                                        # Getting the label
  image = X_test[index].reshape(300,300,3)                                      # Reshaping the image to 300,300,3
  (h, w) = X_test[index].shape[:2]                                              # Obtaining the height and width of the image

  # scale the predicted bounding box coordinates based on the image dimensions so that the box fits correctly on the image
  startX = int(startX*w )
  startY = int(startY*h )
  endX = int(endX *w)
  endY = int(endY*h)
  print((startX, endX,startY, endY))

  # Draw the predicted bounding box on the image
  cv2.rectangle(image, (startX,startY),(endX,endY), color=(255,0,0), thickness=2)
  plt.title("Image corresponds to: " + str(label),{'fontsize':15})              # Print the labels as the title of the image
  plt.imshow(image)

# %%
plot_predictions(55)

# %%
plot_predictions(66)

# %%
plot_predictions(99)

# %%
plot_predictions(255)

# %% [markdown]
# We observe that the bounding boxes are not always predicted to the localization level we would expect from a high-quality object detection system. This may be the reason for the relatively low classification accuracy of the CNN in this object detection problem.

