// written by André Betz 
// http://www.andrebetz.de

/* Hopfield Net */
#include<stdio.h>	/* printf() */
#include<stdlib.h>	/* malloc(), free() */  

class Hopfield
{
private:
	int **m_Mat;
	int  *m_In;
	int  *m_Out;
	int   m_long;
public:
	Hopfield(int length);
	~Hopfield();
	void Init();
	void Learn();
	int Calc();
	void Input(char *Dat);
	void Output(char *Dat);
};

Hopfield::Hopfield(int length)
{
	int i;
	m_long = length;
	
	m_Mat = (int**)malloc(sizeof(int*)*m_long);
	for(i=0;i<m_long;i++)
	{
		m_Mat[i] = (int*)malloc(sizeof(int)*m_long);
	}	
	
	m_In  = (int*)malloc(sizeof(int)*m_long);
	m_Out = (int*)malloc(sizeof(int)*m_long);
	
	Init();
}

Hopfield::~Hopfield()
{
	int i;
	
	free(m_Out);
	free(m_In);
	for(i=0;i<m_long;i++)
	{
		free(m_Mat[i]);
	}
	free(m_Mat);
}

void Hopfield::Init()
{
	int i,j;
	for(i=0;i<m_long;i++)
	{
		for(j=0;j<m_long;j++)
		{
			m_Mat[i][j] = 0;	
		}
	}
}

void Hopfield::Learn()
{
	int i,j;

	for(i=0;i<m_long;i++)
	{
		for(j=0;j<m_long;j++)
		{
			if(i!=j) m_Mat[i][j] += m_In[i] * m_In[j];
		}
	}
}

int Hopfield::Calc()
{
        int i,j,h,end=1;                                            
                                                                     
	for(i=0;i<m_long;i++)                               
	{                                                    
		h = 0;                                       
                for(j=0;j<m_long;j++)                       
                {                                            
                	h += m_In[j] * m_Mat[i][j];                 
                }                                            
                                                                     
                if(h<0) m_Out[i] = -1;                           
                else    m_Out[i] =  1;                           
	}                                                    
                                                                     
        for(i=0;i<m_long;i++)                               
        {                                                    
		for(j=0;j<m_long;j++)                       
                {                                            
                	if(m_Out[i]!=m_In[i])  end = 0;            
                        m_In[i] = m_Out[i];                         
                }                                            
        }                                                    
	
	return end;                                                            
}

void Hopfield::Input(char *Dat)
{
	int i;
	
	for(i=0;i<m_long;i++)
	{
		if(Dat[i]=='.') m_In[i] = -1;
		else		m_In[i] =  1;
	}
}

void Hopfield::Output(char *Dat)
{
	int i;

	for(i=0;i<m_long;i++)
	{
		if(m_Out[i]<0)	Dat[i] = '.';
		else		Dat[i] = 'O';;
	}	
}

void Output(char *Dat,int x)
{
	int i,j;
	for(i=0;i<x;i++)
	{
		for(j=0;j<x;j++)
		{
			printf("%c",Dat[i*x+j]);
		}
		printf("\n");
	}
	printf("\n");	
}

void main()
{	
      	char learnT[] = "OOO."
			".O.."
			".O.."
			".O..";
	
	char learnL[] =	"O..."
			"O..."
			"O..."
			"OOO.";
	
	char misc1[] =	"O.O."
			"...."
			"...."
			"....";

	char misc2[] =	"O..."
			"...."
			"...."
			"..O.";	                         	
                                                                     
	char out[16];
	
	Hopfield net(16);
	
	printf("learn picture\n");	
	net.Input(learnT);
	net.Learn();
	Output(learnT,4);

	net.Input(learnL);
	net.Learn();
	Output(learnL,4);
	
	printf("\ndisturbed picture:\n");
	Output(misc1,4);
	Output(misc2,4);

	printf("\nrecognize picture:\n");	
	net.Input(misc1);
	while(!net.Calc());
	net.Output(out);
	Output(out,4);
	
	net.Input(misc2);
	while(!net.Calc());
	net.Output(out);
	Output(out,4);
}
