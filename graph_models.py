from typing import List, Optional

from pydantic import BaseModel, Field


class Location(BaseModel):
    """Represents a geographic location node."""

    city: Optional[str] = Field(None, description="City name.")
    state: Optional[str] = Field(None, description="State or province.")
    country: Optional[str] = Field(None, description="Country name.")


class Person(BaseModel):
    """Represents a person node e.g. CEO, Founder."""

    name: str = Field(..., description="Full name of the person.")
    role: Optional[str] = Field(
        None, description="Role within the organization e.g. CEO, Founder, CTO."
    )


class Product(BaseModel):
    """Represents a product or service node offered by the organization."""

    name: str = Field(..., description="Name of the product or service.")
    category: Optional[str] = Field(
        None, description="Category or type of the product/service."
    )
    description: Optional[str] = Field(
        None, description="Brief description of the product/service."
    )


class Subsidiary(BaseModel):
    """Represents a subsidiary or related company node."""

    name: str = Field(..., description="Name of the subsidiary company.")
    domain: Optional[str] = Field(
        None, description="Website domain of the subsidiary."
    )


class Industry(BaseModel):
    """Represents an industry or sector node."""

    name: str = Field(
        ..., description="Name of the industry or sector e.g. Beverages, Technology."
    )


class OrganizationGraph(BaseModel):
    """
    Represents an organization and all extractable relationships
    from its website.
    """

    # Subject (the organization itself)
    organization_name: str = Field(
        ..., description="Legal or commonly known name of the organization."
    )
    domain: str = Field(
        ..., description="Website domain of the organization e.g. coca-cola.com."
    )

    # Core attributes (data properties)
    description: Optional[str] = Field(
        None, description="Brief description or mission statement of the organization."
    )
    founded_year: Optional[str] = Field(
        None, description="Year the organization was founded. Use yyyy format."
    )
    employee_count: Optional[str] = Field(
        None,
        description="Approximate number of employees e.g. '10,000+' or '500-1000'.",
    )
    stock_ticker: Optional[str] = Field(
        None,
        description="Stock ticker symbol if publicly traded e.g. KO, AAPL.",
    )

    # Relationships (edges / triples)
    headquarters: Optional[Location] = Field(
        None,
        description=(
            "Primary headquarters location. Triple: "
            "(Org) -[HAS_HEADQUARTERS]-> (Location)."
        ),
    )
    operating_locations: Optional[List[Location]] = Field(
        None,
        description=(
            "Countries or regions the org operates in. Triple: "
            "(Org) -[OPERATES_IN]-> (Location)."
        ),
    )
    key_people: Optional[List[Person]] = Field(
        None,
        description=(
            "Key executives or founders. Triple: "
            "(Org) -[HAS_PERSON]-> (Person)."
        ),
    )
    products_services: Optional[List[Product]] = Field(
        None,
        description=(
            "Products or services offered. Triple: "
            "(Org) -[OFFERS]-> (Product)."
        ),
    )
    industries: Optional[List[Industry]] = Field(
        None,
        description=(
            "Industries the org belongs to. Triple: "
            "(Org) -[BELONGS_TO]-> (Industry)."
        ),
    )
    subsidiaries: Optional[List[Subsidiary]] = Field(
        None,
        description=(
            "Subsidiaries or sister companies. Triple: "
            "(Org) -[HAS_SUBSIDIARY]-> (Subsidiary)."
        ),
    )
    parent_company: Optional[str] = Field(
        None,
        description=(
            "Parent company name if this org is a subsidiary. Triple: "
            "(Org) -[IS_SUBSIDIARY_OF]-> (ParentOrg)."
        ),
    )
    partners: Optional[List[str]] = Field(
        None,
        description=(
            "Known partner organizations. Triple: "
            "(Org) -[PARTNERS_WITH]-> (Org)."
        ),
    )
    competitors: Optional[List[str]] = Field(
        None,
        description=(
            "Known competitor organizations. Triple: "
            "(Org) -[COMPETES_AGAINST]-> (Org)."
        ),
    )
    major_customers: Optional[List[str]] = Field(
        None,
        description=(
            "Other companies explicitly named as major customers or buyers of this "
            "organization's products or services (e.g. foundry customer, OEM client). "
            "Use the name as written in the text. Triple: (Org) -[HAS_MAJOR_CUSTOMER]-> (Org)."
        ),
    )
    key_suppliers: Optional[List[str]] = Field(
        None,
        description=(
            "Other companies explicitly named as suppliers to this organization. "
            "Triple: (Org) -[HAS_SUPPLIER]-> (Org)."
        ),
    )

