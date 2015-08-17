Eigenstyle
======
Principal Component Analysis and Fashion

###To Use

- Find a bunch of images (I used images of dresses from Amazon).
- Put the ones that match your style in the "like" folder, and the others in the "dislike" folder
- In terminal, run 
```bash
python visuals.py
```

###Results

You'll see the principal components in the "eigendresses" folder (examples shown are from my dataset; yours will be different).

![Eigendress](http://graceavery.com/eigenstyle/4_eigendress.png)![Eigendress](http://graceavery.com/eigenstyle/0_eigendress.png)!

In the "history" folder, you'll see a known dress being rebuilt from its components.

![Dress from one component](http://graceavery.com/eigenstyle/dress_763_1.png)![Dress from four components](http://graceavery.com/eigenstyle/dress_763_4.png)![Dress from ten components](http://graceavery.com/eigenstyle/dress_763_10.png)![Dress from fifteen components](http://graceavery.com/eigenstyle/dress_763_15.png)![Dress from thirty components](http://graceavery.com/eigenstyle/dress_763_30.png)![Dress from seventy components](http://graceavery.com/eigenstyle/dress_763_70.png)

In the "recreatedDresses" folder, you can see just the end product of this process for different dresses.

![Original](http://graceavery.com/eigenstyle/6_original.png)![Recreated](http://graceavery.com/eigenstyle/6_recreated.png)

In the "notableDresses" folder, you'll see the prettiest dresses, the ugliest dresses, the most extreme dresses (those that had high scores on many components), etc.

![Prettiest 1](http://graceavery.com/eigenstyle/prettiest_pretty_1.png)![Ugliest 2](http://graceavery.com/eigenstyle/ugliest_ugly_2.png)


In the "createdDresses" folder, you'll find completely new dresses that were made from choosing random values for the principal components.

![New Dress](http://graceavery.com/eigenstyle/RandomDress5.png)![New Dress](http://graceavery.com/eigenstyle/RandomDress18.png)


### More Info
[Blog post](http://blog.thehackerati.com/post/126701202241/eigenstyle)

[Joel Grus's blog post](http://joelgrus.com/2013/06/24/t-shirts-feminism-parenting-and-data-science-part-2-eigenshirts/)
