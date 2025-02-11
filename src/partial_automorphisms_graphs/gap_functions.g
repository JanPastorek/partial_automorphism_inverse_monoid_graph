# Function to check whether two elements are compatible based on your provided logic
IsCompatible := function(perm1, perm2)
    local i_1, i_2;
    
    # Calculate i_1 = perm1.inverse() * perm2
    i_1 := perm1^-1 * perm2;
    
    # Calculate i_2 = perm1 * perm2^-1
    i_2 := perm1 * perm2^-1;
    
    # Check if both i_1 and i_2 are idempotent (idempotent: p * p = p)
    if i_1 * i_1 = i_1 and i_2 * i_2 = i_2 then
        return true;
    else
        return false;
    fi;
end;

# Function to get all rank 1 elements using GreensDClassOfElementNC
GetRank1Elements := function(S)
    local p, D, rank1Elems;
    
    # Define the rank 1 element as a partial permutation
    p := PartialPerm([1], [1]);
    
    # Get the Green's D-class of the element
    D := GreensDClassOfElement(S, p);
    
    # Convert the D-class to a set
    rank1Elems := AsSet(D);
    
    return rank1Elems;
end;

# Function to compute the compatibility matrix for a list of elements
ComputeCompatibilityMatrix := function(rank1Elems)
    local n, matrix, i, j;
    
    n := Length(rank1Elems);
    matrix := [];  # Initialize an empty matrix
    
    # Loop through pairs of elements and fill the matrix
    for i in [1..n] do
        matrix[i] := [];
        for j in [1..n] do
            if IsCompatible(rank1Elems[i], rank1Elems[j]) then
                matrix[i][j] := 1;  # Compatible
            else
                matrix[i][j] := 0;  # Not compatible
            fi;
        od;
    od;
    
    return matrix;
end;

# Function to check if a subset of indices corresponds to a transitive subset
IsTransitiveSubsetMatrix := function(subsetIndices, matrix)
    local i, j;
    
    
    # Check every pair of indices in the subset
    for i in [1..Length(subsetIndices)] do
        for j in [i+1..Length(subsetIndices)] do
            if matrix[subsetIndices[i]][subsetIndices[j]] = 0 then
                return false;  # If any pair is not compatible, it's not transitive
            fi;
        od;
    od;
    
    return true;  # If all pairs are compatible, the subset is transitive
end;

# Function to compute the join of a subset of partial permutations
ComputeJoin := function(subset)
    local joinSubset, i;
    
    if Length(subset) = 1 then
        return subset[1];  # If there's only one element, return it
    fi;
    
    joinSubset := subset[1];
    for i in [2..Length(subset)] do
        joinSubset := JoinOfPartialPerms(joinSubset, subset[i]);  # Join all elements
    od;
    
    return joinSubset;
end;

# Main function to find all transitive subsets of S using the compatibility matrix
IsPAUT := function(S)
    local rank1Elems, matrix, n, indices, subsets, subsetIndices, transitiveSubsets, subset, joinSubset, iterators, iterator, i, j, DClasses, DClass, idempotents, A, E_S, num_nodes, it, HC, HClass, pa, pa_list, partialPerm, partialPermDClass;

    # is semigroup boolean and fundamental ? 
    num_nodes := RankOfPartialPermSemigroup(S);
    idempotents := Idempotents(S);

    A := Semigroup(idempotents);
    E_S := Semigroup(Idempotents(SymmetricInverseSemigroup(num_nodes)));

    if not IsIsomorphicSemigroup(A, E_S) then
        return 1;
    fi;
    
    # DClasses := List(GreensDClassOfElement(S, PartialPerm([],[])));
    DClasses := [];

    for i in Combinations([1..num_nodes], 2) do
        DClass := GreensDClassOfElement(S, PartialPerm([i[1], i[2]], [i[1], i[2]]));
        # return DClass;
        if not DClass in DClasses then
            # H-classes of the height 2 D-classes of S are nontrivial.
            HC := HClasses(DClass);
            # return HC;
            for HClass in HC do
                if Size(HClass) < 2 then
                    return 2;
                fi;
            od;

            Add(DClasses, DClass);

            # at most two D-classes of height 2
            if Length(DClasses) > 2 then
                return 3;
            fi;
        fi;
    od;


    # Get all rank 1 elements using GreensDClassOfElement
    rank1Elems := GetRank1Elements(S);
    

    
    # Compute the compatibility matrix
    matrix := ComputeCompatibilityMatrix(rank1Elems);
    

    n := Length(rank1Elems);
    
    # Filter transitive subsets using the compatibility matrix  
    for subsetIndices in EnumeratorOfCombinations([1..n]) do
        if Length(subsetIndices) > 1 and IsTransitiveSubsetMatrix(subsetIndices, matrix) then

            subset := List(subsetIndices, i -> rank1Elems[i]);
            
            # Compute the join of the subset
            joinSubset := ComputeJoin(subset);
            
            # Check if the join is in S
            if joinSubset in S then
                
                # check all pairs of elements in the subset if they are in S
                for i in [1..Length(subset)] do
                    for j in [i+1..Length(subset)] do
                        if not JoinOfPartialPerms(subset[i], subset[j]) in S then
                            return 4;
                        fi;
                    od;
                od;
                Print("Valid transitive subset: ", subset, "\n");
            fi;

        fi;
    od; 

    # take all elements of the D-classes of height 2
    partialPerm := [];
    for DClass in DClasses do
        partialPermDClass := [];
        for pa in DClass do
            pa_list := [];
            Add (pa_list, DomainOfPartialPerm(pa));
            Add (pa_list, ImageListOfPartialPerm(pa));
            Add(partialPermDClass, pa_list);
        od;
        Add(partialPerm, partialPermDClass);
    od;

    return partialPerm;

end;
