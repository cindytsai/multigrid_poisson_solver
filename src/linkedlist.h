#ifndef LINKEDLIST_H
#define LINKEDLIST_H

class ListNode{
private:

	int N;			// The grid size N at this level
	double *U;		// The solution at the perspective of current level
	double *F;		// The source at the perspective of current level
	double *D;		// The residual at the perspective of current level

	ListNode *nextNode;	// Pointer to the next level
	ListNode *prevNode;	// Pointer to the previous level

	// For smoothing Node
	int step;						// Number of step to do smoothing
	double smoothingError;			// Error after "step" of smoothing

	// For Exact Solver Node
	int option;						// Choose the exact solver
	double target_error;			// The target error for the exact solver

public:
	ListNode(int n);	// Constructor, initialize the node
	
	friend class LinkedList;
};

class LinkedList{
private:
	ListNode *firstNode;	// Pointer to the first node
	ListNode *lastNode;		// Pointer to the last node
	double L;				// Length of the interest region
	double min_x, min_y;	// The lower left point of the region

public:
	LinkedList(): firstNode(0), lastNode(0) {};
	~LinkedList();
	void Push_back(int n);
	void Remove_back();
	void Set_Problem(double l, double o_x, double o_y);	// Set interest region of the problem
	void Set_init(int r);
	double Get_L();
	double* Get_U();					// Get U of the lastNode
	double* Get_D();					// Get D of the lastNode
	double* Get_F();					// Get F of the lastNode
	int Get_N();						// Get grid size N of the lastNode
	double* Get_ptr_smoothingError();	// Get the address of smoothingError of the lastNode
	int* Get_ptr_step();				//	Get the address of step of the lastNode
	int Get_prev_N();					// Get the N at the previous node of the lastNode
	bool Is_firstNode();
};

#endif
