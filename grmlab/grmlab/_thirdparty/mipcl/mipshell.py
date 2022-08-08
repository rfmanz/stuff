from grmlab._thirdparty.mipcl import mipcl

"""Python part of MIPCL-PY."""

_license__ = "GLPL"
__docformat__ = 'reStructuredText'


REAL     = 0x00000000
"""int: The type of real variables that may take any real value."""

INT      = 0x00001000
"""int: The type of integer variables that may take any integer value."""

BIN      = 0x00002000
"""int: The type binary variables that takes only two values, 0 or 1."""

PRIO_MIN = -50
"""int: Minimal priority of a variable."""

PRIO_MAX = 50
"""int: Maximal priority of a variable."""

INF = 1.0e20
"""float: \"Infinity\" value."""

VAR_INF  = 1.0e+12
"""float: \"Infinity\" value for variables.

If a variable is unbounded from above, its upper bound is set to ``VAR_INF``.

Similarly, If a variable is unbounded from below, its lower bound is set to ``-VAR_INF``.
"""

ZERO = 1.0e-12
"""float: Value of \"zero\", i.e., cooefficients that less than ``ZERO`` are treated as zeroes."""

SOS1 = 0x00008000
"""(unsigned) int: Flag indicating that some inequality is a SOS1-constraint. 

Any inequality ``sum(a(j)*x(j): j=1,...,n) <= b`` with all variables having zero lower bounds
can be declared as a SOS1-constraint, which means that at most one of its variables can take
a non-zero value. If a SOS1-constraint has more than one variable of positive current value,
MIPCL applies a special branching procedure to make the constraint feasible.  
"""  

class Var:
    """In MIPCL-PY models, objects of type ``Var`` represent variables.

    Attributes:
        name (str): name, which is used when printing a solution;
        lb (float): lower bound;
        ub (float): upper bound;
        val (float): value (it is computed only if a solution has been found);
        hd (int): handle;
        type (int): type.

    """
    def __init__(self,name="",type=REAL,lb=0.0,ub=VAR_INF,priority=0):
        """The constructor that creates a new variable, and adds that variable to a specified problem.

        Args:
            name (str): name, which is used when printing a solution;
            type (int): type (:data:`REAL`, :data:`INT`, or :data:`BIN`) of variable being created;
            lb (float): lower bound;
            ub (float): upper bound;
            priority (int): priority of variable.

        """
        self.name = name
        self.priority = priority
        self.type = type
        if type & BIN:
            self.lb, self.ub = 0.0, 1.0
        else:
            self.lb, self.ub = lb, ub
        self.hd = Problem.curProb.addVar(self)

    def __repr__(self):
        """Compute a string representation of the ``self`` object.

        Returns:
            str: string representation of the ``self`` object.

        """
        if self.type & BIN:
            s = repr(self.name)
            if self.ub - self.lb > 0.5:
                s += " in {0,1}"
            else:
                s += " = " + repr(self.ub)
        else:
            if self.lb > -VAR_INF+1.0:
                l = repr(self.lb)
            else:
                l = "-inf"
            if self.ub < VAR_INF-1.0:
                u = repr(self.ub)
            else:
                u = "inf"
            if self.type & INT:
                s = l + " <= " + repr(self.name) + " <= " + u + ": integer"
            else:
                s = l + " <= " + repr(self.name) + " <= " + u
        return s

######  Operators
    def __rmul__(self,a):
        """Compute the product of the ``self`` variable and a constant. 

        Args:
	    a (float): multiplier.

	Returns:
            Term: linear term ``a * self``.

        """
        return Term(a,self)

    def __add__(self,lsum):
        """Compute the sum of an :class:`~mipshell.LinSum` object and  the ``self`` variable. 

        Args:
	    lsum: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            LinSum: liner sum ``lsum + self``.

        """
        if isinstance(lsum,LinSum):
            ls = lsum + Term(1.0,self)
        else:
            ls = LinSum([Term(1.0,self)]) + lsum
        return ls

    def __radd__(self,lsum):
        """Compute the sum of the ``self`` variable and an :class:`~mipshell.LinSum` object. 

        Args:
	    lsum: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            LinSum: linear sum ``self + lsum``.

        """
        return LinSum([Term(1.0,self)]) + lsum

    def __sub__(self,lsum):
        """Compute the difference of the ``self`` variable and an :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number. 

        Args:
	    lsum: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            LinSum: linear sum ``self - lsum``.

        """
        if isinstance(lsum,LinSum):
            ls = lsum.__rsub__(Term(1.0,self))
        else:
            ls = LinSum([Term(1.0,self)]) - lsum
        return ls

    def __rsub__(self,lsum):
        """Compute the difference of an :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number and the ``self`` variable. 

        Args:
	    lsum: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            LinSum: linear sum ``lsum - self``.

        """
        return LinSum([Term(-1.0,self)]) + lsum
        return ls

    def __neg__(self):
        """Negate the ``self`` variable. 

	Returns:
            Term: linear term ``-1.0*self``.

        """
        return Term(-1.0,self)

    def __pos__(self):
        """Convert the ``self`` variable into a :class:`~mipshell.Term` object. 

	Returns:
            Term: linear term ``1.0*self``.

        """
        return Term(1.0,self)

    def __le__(self,rhs):
        """Process the inequality ``self <= rhs``.

        If ``rhs`` is a number, the *upper* bound of the ``self`` variable is set to ``rhs``;
        otherwise, a :class:`~mipshell.Ctr` object is created that represents the inequality ``self <= rhs``.    

        Args:
	    rhs: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            Ctr: constraint ``self <= rhs`` if ``rhs`` is not a number,
            or nothing, otherwise. 

       """
        if isinstance(rhs,LinSum) or isinstance(rhs,Term) or isinstance(rhs,Var):
            ctr = Ctr(self-rhs,-INF,0.0)
        else: # rhs is a number
            ctr = self
            if self.ub > rhs:
                self.ub = rhs
        return ctr

    def __ge__(self,rhs):
        """Process the inequality ``self >= lhs``.

        If ``lhs`` is a number, the *lower* bound of the ``self`` variable is set to ``lhs``;
        otherwise, a :class:`~mipshell.Ctr` object is created that represents the inequality ``self >= lhs``.    

        Args:
	    lhs: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            Ctr: constraint ``self >= lhs`` if ``lhs`` is not a number, or nothing, otherwise. 

       """
        if isinstance(rhs,LinSum) or isinstance(rhs,Term) or isinstance(rhs,Var):
            ctr = Ctr(self-rhs,0.0,INF)
        else:
            ctr = self
            if self.lb < rhs:
                self.lb = rhs
        return ctr

    def __eq__(self,rhs):
        """Process the equality ``self == rhs``.

        If ``rhs`` is a number, both *lower* and *upper* bounds of the ``self`` variable are set to ``rhs``;
        otherwise, a :class:`~mipshell.Ctr` object is created that represents the inequality ``self == rhs``.    

        Args:
	    rhs: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            Ctr: constraint ``self == rhs`` if ``lhs`` is not a number, or nothing, otherwise. 

       """
        if isinstance(rhs,Function):
            rhs.y = self
            return rhs
        elif isinstance(rhs,LinSum) or isinstance(rhs,Term) or isinstance(rhs,Var):
            ctr = Ctr(self-rhs,0.0,0.0)
        else:
            ctr = self
            self.lb = rhs
            self.ub = rhs
        return ctr

#######################################################
class Term:
    """``Term`` objects represent linear terms.

    A *linear term*, ``a * var``, is the product of a constant ``a`` and
    a variable ``var``, which is an object of class :class:`~mipshell.Var`.
    In **MIPCL-PY** applications linear terms are always behind the scene.

    Attributes:
        coeff (float): numeric coefficient;
        var (Var): reference to a var object.

    """
    def __init__(self,coeff,var):
        """Initialize a newly created :class:`~mipshell.Term` object to represent ``coeff * var``.

        Args:
           coeff (float): numeric coefficient;
           var (Var): reference to the created :class:`~mipshell.Var` object.

        """
        self.coeff=coeff
        self.var=var

    def __repr__(self):
        """Compute a string representation of the ``self`` object.

        Returns:
            str: string representation of the ``self`` object.

        """
        if self.var is not None:
            s = '{:.4f}*{!r}'.format(self.coeff,self.var.name)
        else:
            s = '{:.4f}'.format(self.coeff)
        return s

    def __neg__(self):
        """Change the sign of the ``self.coeff``. 

	Returns:
            Term: reference to ``self``.

        """
        self.coeff = -self.coeff
        return self

    def __pos__(self):
        """Do nothing. 

	Returns:
            Term: reference to ``self``.

        """
        return self

    def __rmul__(self,a):
        """Multiplier ``self.coeff`` by ``a``. 

        Args:
	    a (float): multiplier.

	Returns:
            Term: reference to ``self``.

        """
        self.coeff *= a
        return self

    def __add__(self,lsum):
        """Compute the sum of an :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number with  the ``self`` object. 

        Args:
	    lsum: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            LinSum: linear sum ``lsum + self``.

        """
        if type(lsum) is LinSum:
            ls = lsum + self
        else:
            ls = LinSum([self]) + lsum
        return ls

    def __radd__(self,lsum):
        """Compute the sum of the ``self`` object with an :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

        Args:
	    lsum: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            LinSum: linear sum ``self + lsum``.

        """
        return LinSum([self]) + lsum

    def __sub__(self,lsum):
        """Compute the difference of the ``self`` object and :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number. 

        Args:
	    lsum: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            LinSum: linear sum ``self - lsum``.

        """
        if isinstance(lsum,LinSum):
            ls = lsum.__rsub__(self)
        else:
            ls = LinSum([self]) - lsum
        return ls

    def __rsub__(self,lsum):
        """Compute the difference of an :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number and the ``self`` linear term. 

        Args:
	    lsum: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            LinSum: linear sum ``lsum - self``.

        """
        return lsum - LinSum([self])

    def __le__(self,rhs):
        """Process the inequality ``self <= rhs``.

        If ``rhs`` is a number, the *upper* bound of the ``self`` variable is set to ``rhs / self.coeff``;
        otherwise, a :class:`~mipshell.Ctr` object is created that represents the inequality ``self <= rhs``.    

        Args:
	    rhs: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            Ctr: constraint ``self <= rhs`` if ``rhs`` is not a number,
            or nothing, otherwise. 

       """
        if isinstance(rhs,LinSum) or isinstance(rhs,Term) or isinstance(rhs,Var):
            ctr = Ctr(self-rhs,-INF,0.0)
        else: # rhs is a number
            ctr = self
            if self.coeff > 1.0e-9:
                self.var.__le__(rhs/self.coeff)
            elif self.coeff < -1.0e-9:
                self.var.__ge__(rhs/self.coeff)
        return ctr

    def __ge__(self,rhs):
        """Process the inequality ``self >= lhs``.

        If ``lhs`` is a number, the *lower* bound of the ``self`` variable is set to ``lhs / self.coeff``;
        otherwise, a :class:`~mipshell.Ctr` object is created that represents the inequality ``self >= lhs``.    

        Args:
	    lhs: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            Ctr: constraint ``self >= lhs`` if ``lhs`` is not a number,
            or nothing, otherwise. 

       """
        if isinstance(rhs,LinSum) or isinstance(rhs,Term) or isinstance(rhs,Var):
            ctr = Ctr(self-rhs,0.0,INF)
        else:
            ctr = self
            if self.coeff > 1.0e-9:
                self.var.__ge__(rhs/self.coeff)
            elif self.coeff < -1.0e-9:
                self.var.__le__(rhs/self.coeff)
        return ctr

    def __eq__(self,rhs):
        """Process the equality ``self == rhs``.

        If ``rhs`` is a number, both *lower* and *upper* bounds of the ``self`` variable are set to ``rhs / self.coeff``;
        otherwise, a :class:`~mipshell.Ctr` object is created that represents the inequality ``self == rhs``.    

        Args:
	    rhs: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            Ctr: constraint ``self == rhs`` if ``rhs`` is not a number,
            or nothing, otherwise. 

       """
        if isinstance(rhs,LinSum) or isinstance(rhs,Term) or isinstance(rhs,Var):
            ctr = Ctr(self-rhs,0.0,0.0)
        else:
            ctr = self
            if (abs(self.coeff) > 1.0e-9):
                self.var.__eq__(rhs/self.coeff)
        return ctr

class LinSum:
    """``LinSum`` objects represent linear function.

    Each ``~mipshell.LinSum`` object is just a list of :class:`~mipshell.Term` objects.

    Attributes:
        terms (list of Term): linear sum.

    """
    def __init__(self,terms = []):
        """Initialize a newly created ``~mipshell.LinSum`` object with a list of :class:`~mipshell.Term` objects.

        Args:
           terms (list of Term): linear sum.

        """
        self.term = terms

    def __repr__(self):
        """Compute a string representation of the ``self`` object.

        Returns:
            str: string representation of the ``self`` object.

        """
        k = 0
        s = ''
        for t in self.term:
            if (k and t.coeff >= 0.0):
                s += '+'
            s += repr(t)
            k += 1
        return s

    def __neg__(self):
        """Change the signs of coefficients in all terms the ``self`` linear sum. 

	Returns:
            LinSum: reference to ``self``.

        """
        for t in self.term:
            t.coeff = -t.coeff
        return self

    def __pos__(self):
        """Do nothing. 

	Returns:
            LinSum: reference to ``self``.

        """
        return self

    def __rmul__(self,a):
        """Multiplies the coefficients by ``a`` in all terms of the ``self`` object. 

        Args:
	    a (float): multiplier.

	Returns:
            LinSum: reference to ``self``.

        """
        for t in self.term:
            t.coeff *= a
        return self

    def __add__(self,lsum):
        """Compute the sum of :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number with the *self* object. 

        Args:
	    lsum: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            LinSum: linear sum ``lsum + self``.

        """
        if isinstance(lsum,LinSum):
            self.term.extend(lsum.term)
            del lsum
        elif isinstance(lsum,Term):
            self.term.append(lsum)
        elif isinstance(lsum,Var):
            self.term.append(Term(1.0,lsum))
        else:
            self.term.append(Term(lsum,None))
        return self     

    def __radd__(self,lsum):
        """Compute the sum of the ``self`` object with an :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

        Args:
	    lsum: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            LinSum: linear sum ``self + lsum``.

        """
        if isinstance(lsum,Term):
            self.term.append(lsum)
        elif isinstance(lsum,Var):
            self.term.append(Term(1.0,lsum))
        else:
            self.term.append(Term(lsum,None))
        return self

    def __sub__(self,lsum):
        """Compute the difference of the ``self`` object and an :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number. 

        Args:
	    lsum: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            LinSum: linear sum ``self - lsum``.

        """
        if isinstance(lsum,LinSum):
            for t in lsum.term:
                t.coeff=-t.coeff
                self.term.append(t)
            del lsum
        elif isinstance(lsum,Term):
            lsum.coeff=-lsum.coeff
            self.term.append(lsum)
        elif isinstance(lsum,Var):
            self.term.append(Term(-1.0,lsum))
        else:
            self.term.append(Term(-lsum,None))
        return self

    def __rsub__(self,lsum):
        """Compute the difference of an :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`Var' object, or a number and the ``self`` linear term. 

        Args:
	    lsum: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            LinSum: linear sum ``lsum - self``.

        """
        for t in self.term:
            t.coeff = -t.coeff
        if isinstance(lsum,Term):
            self.term.append(lsum)
        elif isinstance(lsum,Var):
            self.term.append(Term(1.0,lsum))
        else:
            self.term.append(Term(lsum,None))
        return self

    def __le__(self,rhs):
        """Process the inequality ``self <= rhs``.

        If ``rhs`` is a number, the constrait (:class:`~mipshell.Ctr` object) ``self <= rhs`` is created;
        otherwise, a :class:`~mipshell.Ctr` object is created that represents the inequality ``self - rhs <= 0``.    

        Args:
	    rhs: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            Ctr: constraint ``self <= rhs``. 

        """
        if isinstance(rhs,LinSum) or isinstance(rhs,Term) or isinstance(rhs,Var):
            ctr = Ctr(self-rhs,-INF,0.0)
        else:
            ctr = Ctr(self,-INF,rhs)
        return ctr

    def __ge__(self,lhs):
        """Process the inequality ``self >= lhs``.

        If ``lhs`` is a number, the constrait (:class:`~mipshell.Ctr` object) ``self >= lhs`` is created;
        otherwise, a :class:`~mipshell.Ctr` object is created that represents the inequality ``self - lhs >= 0``.    

        Args:
	    rhs: :class:`~mipshell.LinSum`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            Ctr: constraint ``self >= lhs``. 

        """
        if isinstance(lhs,LinSum) or isinstance(lhs,Term) or isinstance(lhs,Var):
            ctr = Ctr(self-lhs,0.0,INF)
        else:
            ctr = Ctr(self,lhs,INF)
        return ctr

    def __eq__(self,rhs):
        """Process the equality ``self == rhs``.

        If ``rhs`` is a number, the constrait (:class:`~mipshell.Ctr`` object) ``self == rhs`` is created;
        otherwise, a :class:`~mipshell.Ctr` object is created that represents the inequality ``self - rhs == 0``.    

        Args:
	    rhs: :class:`LinSum:class:`, :class:`~mipshell.Term`, :class:`~mipshell.Var` object, or a number.

	Returns:
            Ctr: constraint ``self == rhs``. 

        """
        if isinstance(rhs,LinSum) or isinstance(rhs,Term) or isinstance(rhs,Var):
            ctr = Ctr(self-rhs,0.0,0.0)
        else:
            ctr = Ctr(self,rhs,rhs)
        return ctr

###########################################################
class Ctr:
    """A ``Ctr`` object is a linear sum (:class:`~mipshell.LinSum` object) with lower and upper bounds on its values.

    Attributes:
        lsum (LinSum): linear sum (linear function);
        lhs (float): left hand side;
        rhs (float): right hand side;
        hd (int): handle that uniquely identifies the constraint.
    """
    def __init__(self,lsum,lhs=-INF,rhs=INF,ctrType=0):
        """Initialize a newly created :class:`~mipshell.Ctr` object.

        Args:
           lsum (LinSum): linear sum (function);
           lhs (float): left hand side;
           rhs (float): right hand side;
           ctrType (int): constraint type.

        """
        self.lsum=lsum
        self.lhs=lhs
        self.rhs=rhs
        self.type=ctrType
        self.hd = Problem.curProb.addCtr(self)

    def _repr(self,left=True,right=True):
        """Compute a string representation of the ``self`` object.

        Args:
           left (bool): if ``True``, the left hand side part of constraint ``self`` is represented by the return string;
           right (bool): if ``True``, the right hand side part of constraint ``self`` is represented by the return string.

        Returns:
            str: string representation of the ``self`` object.

        """
        if self.rhs-self.lhs < 1.0e-10:
            s = repr(self.lsum) + ' == {:.4f}'.format(self.rhs)
        else:
            if left and self.lhs > -INF+1.0:
                s = '{:.4f} <= '.format(self.lhs)
            else:
                s = ''
            s+= repr(self.lsum)
            if right and self.rhs < INF-1.0:
                s+= ' <= {:.4f}'.format(self.rhs)
        return s
    
    def __repr__(self):
        """Compute a string representation of the ``self`` object.

        Returns:
            str: string representation of the ``self`` object.

        """
        return self._repr()

    def __le__(self,rhs):
        """Change the right hand side of the ``self`` constraint.

        Args:
	    rhs (float): new right hand side value.

	Returns:
            Ctr: reference to ``self``. 

        """
        self.rhs=rhs
        return self

    def __ge__(self,lhs):
        """Change the left hand side of the ``self`` constraint.

        Args:
	    lhs (float): new left hand side value.

	Returns:
            Ctr: reference to ``self``. 

        """
        self.lhs=lhs
        return self

    def setType(self, ctrType):
        """Change the type of the ``self`` constraint.

        Args:
	    ctrType (int): in the current version of MIPCL-PY, ``ctrType=SOS1``;
                           all other values of ``ctrType`` are ignored.

	Returns:
            Ctr: reference to ``self``. 

        """

        if ctrType == SOS1:
            self.type|=SOS1
            Problem.curProb.isPureLP=False
        return self

class Function:
    """The ``Function`` objects represent piece-wise linear functions *y = f(x)*.

    Attributes:
        x (Var): function argument;
        y (Var): function value;
        points (list of pairs (tuples) of float or int): function break points;
        hd (int): handle that uniquely identifies the function constraint.

    """
    def __init__(self,x,points):
        """Initialize a :class:`~mipshell.Function` object.

        Args:
            x (Var): function argument;
            points (list of pairs (tuples) of float or int): function break points.

        Note:
            This constructor initializes just a :class:`~mipshell.Function` object (not a function constraint).
            A function constraint is set by the equality operator (see __eq__ below).

        """
        self.x = x
        self.y = None
        self.points = points
        self.hd = Problem.curProb.addFunc(self)

    def __eq__(self, y):
        """Set a function constrain.

        Args:
            y (Var): variable which value is equal to the value of the function represented by the ``self`` object.

	Returns:
            Function: reference to ``self``.         

        """
        if isinstance(y,Var):
            self.y = y
        else:
            raise TypeError('Variable is required')
        return self

###########################################################
class Problem:
    """This class connects all components --- variables, constraints, objective --- together to build what is forms a (MIP) Problem.
   
    Attributes:
       curProb (Problem): current problem, this class attribute facilitates formulating MIPs;
       sence (bool): if ``True``, the objective is maximized; otherwise, the objective is minimized; 
       obj (LinSum): reference to a ``~mipshell.LinSum`` object;
       isPureLP (bool): if ``True``, the problem is a MIP; otherwise, it is an LP;
       is_solution (bool): when the problem has been solved, this flag is set to ``True`` if a feasible solution has been found;
            otherwise, the flag is set to ``False``;
       is_solutionOptimal (bool): when the problem has been solved, this flag is set to ``True`` if an optimal solution has been found;
            otherwise, the flag is set to ``False``;
       is_infeasible (bool): when the problem has been solved, this flag is set to ``True`` if the problem is infeasible (has no solution);
            otherwise, the flag is set to ``False``;
       is_unbounded (bool): when the problem has been solved, this flag is set to ``True`` if the problem is unbounded
            (its objective tends to (+/-)infinity);
            otherwise, the flag is set to ``False``;
       vars (list of Var): list of problem variables;
       ctrs (list of Ctr): list of problem constraints;
       funcs (list of Function): list of problem function constraints;
       name (str): problem name.

    """
    curProb = None
    def __init__(self,name):
        """Initialize an empty (without variables, constraints, and objective) problem.

        Args:
            name (str): problem name.

        """
        Problem.curProb = self
        self.sense = None
        self.obj = None
        self.isPureLP = True
        self.is_solution = None
        self.is_solutionOptimal = False
        self.is_infeasible = self.is_unbounded = False   
        self.vars = []
        self.ctrs = []
        self.funcs = []
        self.name = name

    def addVar(self,var):
        """Add a variable to the problem list of variables.

        Args:
            var (Var): variable to be added.

        Returns:
            int: handle of added variable.

        """
        self.vars.append(var)
        if var.type & (INT|BIN):
            self.isPureLP = False
        return len(self.vars)-1

    def addCtr(self,ctr):
        """Add a constraint to the problem list of constraints.

        Args:
            ctr (Ctr): constraint to be added.

        Returns:
            int: handle of added constraint.

        """
        self.ctrs.append(ctr)
        if ctr.type & SOS1:
            self.isPureLP = False
        return len(self.ctrs)-1

    def addFunc(self,func):
        """Add a function constraint to the problem list of function constrsints.

        Args:
            func (Function): function constraint to be added.

        Returns:
            int: handle of added function-constraint.

        """
        self.isPureLP = False
        self.funcs.append(func)
        return len(self.funcs)-1

    def activate(self):
        """Set the current problem to be the problem represented by the ``self`` object."""
        curProb = self

    def minimize(self,lsum):
        """Set the objective to be minimized.

        Args:
            lsum (LinSum): linear function.

        """
        self.sense = "min"
        if isinstance(lsum,LinSum):
            self.obj = lsum
        elif isinstance(lsum,Term):
            self.obj = LinSum([lsum])
        elif isinstance(lsum,Var):
            self.obj = LinSum([Term(1.0,lsum)])
        else:
            self.obj = LinSum()

    def maximize(self,lsum):
        """Set the objective to be maximized.

        Args:
            lsum (LinSum): linear function.

        """
        self.sense = "max"
        if isinstance(lsum,LinSum):
            self.obj = lsum
        elif isinstance(lsum,Term):
            self.obj = LinSum([lsum])
        elif isinstance(lsum,Var):
            self.obj = LinSum([Term(1.0,lsum)])
        else:
            self.obj = LinSum()

    def mipclModel(self,silent):
        """ The function initializes either a CLP-instance for the LP problem or a CMIP-instance for the MIP problem.

        Args:
           silent (bool): if set to ``True``, MIPCL-solver will not display any run-time messages.

        """
        n = len(self.vars)
        m = len(self.ctrs)
        nz = 0
        for ctr in self.ctrs:
            if ctr.lsum is not None:
                nz += len(ctr.lsum.term)
        fn = len(self.funcs)
        funcSize = 0
        for func in self.funcs:
            funcSize+=len(func.points)
        if self.isPureLP:
            mp = mipcl.CLP("pythonLP")
            mp.preprocOff()
        else:
            mp = mipcl.CMIP("pythonMIP")
        self.mp = mp
        self.mp.beSilent(silent)
        mp.openMatrix(m+3*fn,n+funcSize,nz+3*funcSize+2*fn,True,False,0,0,0)
        for j, var in enumerate(self.vars):
            mp.addVar(var.hd,var.type,0.0,var.lb,var.ub)
            var.val = 0.0 # is used when adding constraints to handle terms with the same variables
            if not self.isPureLP:
                if var.priority:
                    mp.setVarPriority(j,var.priority)
        hd = n
        for func in self.funcs:
            for i in range(len(func.points)):
                mp.addVar(hd,0,0.0,0.0,1.0)
                hd += 1
        if self.sense == 'max':
            mp.setObjSense(True)
        else:
            mp.setObjSense(False)
        for t in self.obj.term:
            mp.setObjCoeff(t.var.hd,t.coeff)
        row = 0
        for ctr in self.ctrs:
            if ctr.lsum is not None:
                v = 0.0
                for t in ctr.lsum.term:
                    if t.var is None:
                        v += t.coeff
                    else:
                        t.var.val += t.coeff
                lhs = ctr.lhs
                rhs = ctr.rhs
                if v > ZERO or v < -ZERO:
                    if lhs > -INF+1.0:
                        lhs -= v
                    if rhs < INF-1.0:
                        rhs -= v
                mp.addCtr(row,ctr.type,lhs,rhs)
                for t in ctr.lsum.term:
                    if t.var is not None:
                        if t.var.val > ZERO or t.var.val < -ZERO:
                            mp.addEntry(t.var.val,row,t.var.hd)
                            t.var.val = 0.0
                row += 1
# model function constraints; each such constraint is represented by 3 equations
        for func in self.funcs:
            mp.addCtr(row,0,0.0,0.0)
            mp.addEntry(-1.0,row,func.x.hd)
            hd = n
            for point in func.points:
                mp.addEntry(point[0],row,hd)
                hd += 1
            row += 1
            mp.addCtr(row,0,0.0,0.0)
            mp.addEntry(-1.0,row,func.y.hd)
            hd = n
            for point in func.points:
                mp.addEntry(point[1],row,hd)
                hd += 1
            row += 1
            mp.addCtr(row,0x00010000,1.0,1.0) # 0x00010000 is SOS2 flag in CMIP
            for j in range(n,hd):
                mp.addEntry(1.0,row,j)
            row += 1
            n = hd
            
        mp.closeMatrix()
 
    def optimize(self,silent=True,timeLimit=10000000, gap=0.0, solFileName=""):
        """Start solving the problem.

        Args:
            silent (bool): if set to ``True``, no MIPCL info-messages are displayed;
            timeLimit (int): time limit (in seconds) for the solver.

        """
        self.mipclModel(silent)
        self.mp.optimize(timeLimit,gap,solFileName)
        if self.mp.isSolution():
            self.is_solution = True
            self.obj_val=self.mp.getObjVal()
            self.mp.setSolution();
            for var in self.vars:
                var.val=self.mp.getOptVarValue(var.hd)
            if self.isPureLP:
                self.is_solutionOptimal = True
                for ind,ctr in enumerate(self.ctrs):
                    ctr.price = self.mp.getShadowPrice(ind)
            else:
            	self.is_solutionOptimal = self.mp.isSolutionOptimal()
        else:
            self.is_solution = False
            if self.isPureLP:
                if self.mp.isLpInfesible():
                    self.is_infeasible = True
                    self.mp.setCertificate()
                    for var in self.vars:
                        var.val=self.mp.getOptVarValue(var.hd)
                    for ind,ctr in enumerate(self.ctrs):
                        ctr.price = self.mp.getShadowPrice(ind)
                else:
                    self.is_unbounded = True
            else:
                self.is_infeasible = not self.mp.timeLimitStop()

    def getObjVal(self):
        """Get the objective value.

        Returns:
            float: the objective value of the solution found.

        """
        return self.obj_val

    def _print(self):
        """Print the problem given by the ``self`` object.

        Note:
           The procedure is used only debugging.

        """
        print(self.sense + " " + repr(self.obj))
        for ctr in self.ctrs:
            print(ctr)
        for var in self.vars:
            print(var)

    def printSolution(self):
        """ The procedure prints the solution.

            When an LP has no feasible solution, the procudure prints a certificate of inconsistency.

        """
        if self.is_solution is not None:
            if self.is_solution == True:
                print('Optimal objective value is {:.4f}'.format(self.obj_val))
                for var in self.vars:
                    print('{!r} = {:.4f}'.format(var.name,var.val))
                if self.isPureLP == True:
                    print('=== Constraint shadow prices (dual solution):') 
                    for ctr in self.ctrs:
                    	print('{:.4f} : {!r}'.format(ctr.price,ctr))
            else:
                if self.isPureLP: # LP
                    print('=== Problem constraints are contradictory.')
                    print('Proof of inconsistency:')
                    q = 0.0
                    for ctr in self.ctrs:
                        price = ctr.price
                        if price > 1.0e-6:
                            print('{:.4f} x ({!s})'.format(price,ctr._repr(left=False)))
                            q += ctr.rhs * price
                        elif price < -1.0e-6:
                            print('{:.4f} x ({!s})'.format(price,ctr._repr(right=False)))
                            q += ctr.lhs * price

                    for var in self.vars:
                        val = var.val
                        if val > 1.0e-6:
                            print('{:.4f} x ({!r} <= {:.4f})'.format(val,var.name,var.ub))
                            q += var.ub * val
                        elif val < -1.0e-6:
                            print('{:.4f} x ({:.4f} <= {!r})'.format(val,var.lb,var.name))
                            q += var.lb * val
                    print('= (0 <= {:.4f})'.format(q))
                else: # MIP
                    print('Optimal solution has not been found')
        else:
            print('Please call optimize() first')

    def writeSolution(self,solfile):
        """ The procedure writes the solution to a file.

        Args:
           solfile (str): file name. 

        """
        if self.is_solution is not None:
            with open(solfile,'w') as f:
                if self.is_solution == True:
                    f.write('Optimal objective value is {:.4f}'.format(self.obj_val))
                    for var in self.vars:
                        f.write('{!r} = {:.4f}'.format(var.name,var.val))
                    if self.isPureLP == True:
                        f.write('=== Constraint shadow prices (dual solution):') 
                        for ctr in self.ctrs:
                    	    f.write('{:.4f} : {!r}'.format(ctr.price,ctr))
                else:
                    if self.isPureLP: # LP
                        f.write('=== Problem constraints are contradictory.')
                        f.write('Proof of inconsistency:')
                        q = 0.0
                        for ctr in self.ctrs:
                            price = ctr.price 
                            if price > 1.0e-6:
                                f.write('{:.4f} x ({!s})'.format(price,ctr._repr(left=False)))
                                q += ctr.rhs * price
                            elif price < -1.0e-6:
                                f.write('{:.4f} x ({!s})'.format(price,ctr._repr(right=False)))
                                q += ctr.lhs * price

                        for var in self.vars:
                            val = var.val
                            if val > 1.0e-6:
                                f.write('{:.4f} x ({!r} <= {:.4f})'.format(val,var.name,var.ub))
                                q += var.ub * var.val
                            elif val < -1.0e-6:
                                f.write('{:.4f} x ({:.4f} <= {!r})'.format(val,var.lb,var.name))
                                q += var.lb * val
                        f.write('= (0 <= {:.4f})'.format(q))
                    else: # MIP
                        f.write('Optimal solution has not been found')
        else:
            print('Please call optimize() first')

##################### Global functions  ###################

def getCurProblem():
    """
    Returns:
        Problem: currently active problem.

    """ 
    return Problem.curProb

def VarVector(sizes, name, type=REAL, lb=0.0, ub=VAR_INF, priority=0):
    """Create a vector (list) of variables.

    Args:
       sizes (list or tuple of int):
       name (str): vector name that is used when the values of variables are printed;
       type (int): type of vector variables (:data:`REAL`, :data:`INT`, or :data:`BIN`),
                   all vector variables are of the same type;
       lb (float): common lower bound of all vector variables;
       ub (float): common upper bound of all vector variables;
       priority (int): common priority (``in range(-50,51)``) of all vector variables.

    Returns:
       VarVector: reference to the created list of variables.

    :warning: Maximal dimension of a vector is 4!

    **Example of usage**::

        y = VarVector([n],"y") # 1-dimensional vector of size n of non-negative real variables
        x = VarVector([n,m],"x",BIN,priority=2) # 2-dimensional matrix of size (m x n) of binary variables 

    """
    dim = len(sizes)
    vec = []
    if dim == 1:
        vec = [Var(name+'('+str(i)+')',type,lb,ub,priority) for i in range(sizes[0])]
    elif dim == 2:
        vec = [[Var(name+'('+str(i)+','+str(j)+')',type,lb,ub,priority)\
            for j in range(sizes[1])] for i in range(sizes[0])]
    elif dim == 3:
        vec = [[[Var(name+'('+str(i)+','+str(j)+ ','+str(k)+')',type,lb,ub,priority)\
            for k in range(sizes[2])]\
            for j in range(sizes[1])] for i in range(sizes[0])]
    elif dim == 4:
        vec = [[[[Var(name+'('+str(i)+','+str(j)+ ','+str(k)+','+str(l)+')',type,lb,ub,priority)\
            for l in range(sizes[3])] for k in range(sizes[2])]\
            for j in range(sizes[1])] for i in range(sizes[0])]
    else:
        raise IndexError('Vectors of dimension greater than 4 are not allowed')
    return vec
       
def VarArray(indices, name, type=REAL, lb=0.0, ub=VAR_INF, priority=0):
    """Create an array (dictionary) of variables.

    Args:
       indices (list of any printable objects): list of keys;
       name (str): vector name that is used when the values of variables are printed;
       type (int): type of vector variables (:data:`REAL`, :data:`INT`, or :data:`BIN`), all vector variables are of the same type;
       lb (float): common lower bound of all vector variables;
       ub (float): common upper bound of all vector variables;
       priority (int): common priority (``in range(-50,51)``) of all vector variables.

    Returns:
        VarArray: reference to the created array of variables.

    **Example of usage**::

        I = [(1,2,3),'Minsk','Bonn',(3,2,1)]
        x = VarArray(I,"x",BIN,priority=2) # array of binary variables 

    """
    ar = {}
    for ind in indices:
        ar[ind] = Var(name + "(" + str(ind) + ")",type,lb,ub,priority)
    return ar

def sum_(terms):
    """Main tool for writing linear expressions.

    The function is a substitution for the Python ``sum`` operator,
    which returns a numeric value. That is why we cannot use ``sum``
    for writing linear expressions. 

    Returns:
        LinSum: linear sum.

    """
    _terms = []
    for t in terms:
        if isinstance(t,LinSum):
            _terms.extend(t.term)
            del t
        elif isinstance(t,Term):
            _terms.append(t)
        else:
            if isinstance(t,Var):
                _terms.append(Term(1.0,t))
            else:
                _terms.append(Term(t,None))
    del terms
    return LinSum(_terms)

def function(x, points):
    """Alternative way for creating a ``~mipshell.Function`` object.

    Args:
       x (Var): function argument;
       points (list of pairs (tuples) of float or int): function break points.

    Returns:
       Function: piece-wise linear function.

    """
    return Function(x,points)

def minimize(lsum):
    """Call ``minimize`` for the currently active problem.

    Args:
        lsum (LinSum): linear expression to be set as the objective function.

    """ 
    Problem.curProb.minimize(lsum)

def maximize(lsum):
    """Call ``maximize`` for the currently active problem.

    Args:
        lsum (LinSum): linear expression to be set as the objective function.

    """ 
    Problem.curProb.maximize(lsum)

def optimize(silent=True,timeLimit=10000000):
    Problem.curProb.optimize(silent,timeLimit)


